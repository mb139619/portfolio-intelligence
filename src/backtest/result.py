"""
BacktestResult — the serialisable output contract.

Everything downstream consumes only this object. The tearsheet never reaches
back into the engine, never recomputes a metric, never touches the store: if a
number is missing from the result it is added here, not calculated in a
template. That is the same discipline `src/export/` already applies to the
portfolio dashboard, and it is what makes a run reproducible — reload the
directory, re-render, get the identical report without rerunning anything.

`RunMeta` is the part that makes a result trustworthy rather than merely
plausible. A backtest with no provenance is an anecdote: it records the commit
it ran from (and whether the tree was dirty), the cost assumptions, the
estimation windows, the publication lag and the calendar policy. Anyone
comparing two runs needs to know which of those differed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl


def current_git_commit() -> tuple[str, bool]:
    """
    (commit hash, dirty). Returns ("unknown", True) outside a git tree.

    Dirty is recorded rather than ignored: a run from a modified working tree
    cannot be reproduced from its commit, and pretending otherwise is worse
    than admitting it.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return sha, bool(status)
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unknown", True


@dataclass(frozen=True)
class Fold:
    """
    One walk-forward split.

    Train and test must not overlap, and the check is here rather than in the
    harness because an overlapping fold is the most expensive kind of silent
    bug: it produces a better-looking out-of-sample result than the strategy
    earned.
    """

    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        if self.train_end < self.train_start:
            raise ValueError(f"fold {self.index}: train window is inverted")
        if self.test_end < self.test_start:
            raise ValueError(f"fold {self.index}: test window is inverted")
        if self.test_start <= self.train_end:
            raise ValueError(
                f"fold {self.index}: test starts {self.test_start} but training "
                f"runs through {self.train_end}. Overlapping folds leak the "
                f"answer into the exam."
            )

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days

    def contains(self, d: date) -> bool:
        """Is `d` inside this fold's out-of-sample window?"""
        return self.test_start <= d <= self.test_end

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Fold:
        return cls(
            index=int(d["index"]),
            train_start=date.fromisoformat(d["train_start"]),
            train_end=date.fromisoformat(d["train_end"]),
            test_start=date.fromisoformat(d["test_start"]),
            test_end=date.fromisoformat(d["test_end"]),
        )


@dataclass(frozen=True)
class RunMeta:
    """Everything needed to reproduce, or fairly compare, a run."""

    strategy: str
    universe: list[str]
    start: date
    end: date

    strategy_params: dict = field(default_factory=dict)

    # Modelling assumptions — the things that make two runs incomparable.
    calendar: str = "trading_days"
    periods_per_year: int = 252
    publication_lag_days: int = 0
    rebalance: str = "unspecified"
    costs: dict = field(default_factory=dict)
    estimation_windows: dict = field(default_factory=dict)
    calendar_policy: str = "native"

    # Provenance.
    git_commit: str = ""
    git_dirty: bool = False
    created_at: str = ""
    data_snapshot: dict = field(default_factory=dict)

    @classmethod
    def create(cls, **kwargs) -> RunMeta:
        """Build with git provenance and timestamp filled in automatically."""
        sha, dirty = current_git_commit()
        kwargs.setdefault("git_commit", sha)
        kwargs.setdefault("git_dirty", dirty)
        kwargs.setdefault(
            "created_at", datetime.now().isoformat(timespec="seconds")
        )
        return cls(**kwargs)

    @property
    def is_reproducible(self) -> bool:
        return self.git_commit != "unknown" and not self.git_dirty

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RunMeta:
        d = dict(d)
        d["start"] = date.fromisoformat(d["start"])
        d["end"] = date.fromisoformat(d["end"])
        return cls(**d)

    def summary(self) -> str:
        flag = "" if self.is_reproducible else "   [DIRTY TREE - not reproducible]"
        return "\n".join([
            f"-- Run: {self.strategy} --{flag}",
            f"  window       {self.start} -> {self.end}",
            f"  universe     {len(self.universe)} assets",
            f"  calendar     {self.calendar} ({self.periods_per_year}/yr), "
            f"policy {self.calendar_policy}",
            f"  rebalance    {self.rebalance}",
            f"  pub. lag     {self.publication_lag_days}d",
            f"  commit       {self.git_commit[:12] or 'unknown'}",
        ])


@dataclass
class BacktestResult:
    """
    The complete, self-contained output of one run.

    Schemas:
      equity_curve : date | equity | ret
      positions    : date | ticker | weight
      trades       : date | ticker | delta_weight | cost
    """

    equity_curve: pl.DataFrame
    positions: pl.DataFrame
    trades: pl.DataFrame
    metrics: dict
    run_meta: RunMeta
    folds: list[Fold] | None = None

    REQUIRED = {
        "equity_curve": {"date", "equity", "ret"},
        "positions": {"date", "ticker", "weight"},
        "trades": {"date", "ticker", "delta_weight", "cost"},
    }

    def __post_init__(self) -> None:
        for name, needed in self.REQUIRED.items():
            got = set(getattr(self, name).columns)
            missing = needed - got
            if missing:
                raise ValueError(
                    f"{name} is missing column(s) {sorted(missing)}; got {sorted(got)}"
                )

    # ------------------------------------------------------------------
    # Derived views (cheap — anything expensive belongs in `metrics`)
    # ------------------------------------------------------------------

    @property
    def returns(self):
        return self.equity_curve["ret"].to_numpy()

    @property
    def dates(self) -> list[date]:
        return self.equity_curve["date"].to_list()

    @property
    def is_long_short(self) -> bool:
        """
        Did this run ever hold a short?

        Derived from what was actually held rather than from the mandate: a
        long/short configuration that never went short produced a long-only
        record, and the metrics that mean something depend on the record.
        """
        return bool(len(self.positions) and (self.positions["weight"] < 0).any())

    @property
    def gross_exposure(self):
        """Σ|w| per bar. Missing on runs saved before exposures were tracked."""
        if "gross" in self.equity_curve.columns:
            return self.equity_curve["gross"].to_numpy()
        return None

    @property
    def net_exposure(self):
        if "net" in self.equity_curve.columns:
            return self.equity_curve["net"].to_numpy()
        return None

    @property
    def total_cost(self) -> float:
        return float(self.trades["cost"].sum()) if len(self.trades) else 0.0

    @property
    def turnover(self) -> float:
        """Total one-way turnover, as a multiple of average book value."""
        if not len(self.trades):
            return 0.0
        return float(self.trades["delta_weight"].abs().sum())

    def out_of_sample(self) -> pl.DataFrame:
        """
        The equity curve restricted to walk-forward test windows.

        A single undifferentiated curve is not an acceptable deliverable, so
        this is the one that matters when judging a strategy.
        """
        if not self.folds:
            return self.equity_curve
        mask = [any(f.contains(d) for f in self.folds) for d in self.dates]
        return self.equity_curve.filter(pl.Series(mask))

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the run so it can be re-rendered later without recomputation."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        self.equity_curve.write_parquet(d / "equity_curve.parquet")
        self.positions.write_parquet(d / "positions.parquet")
        self.trades.write_parquet(d / "trades.parquet")

        sidecar = {
            "metrics": self.metrics,
            "run_meta": self.run_meta.to_dict(),
            "folds": [f.to_dict() for f in self.folds] if self.folds else None,
        }
        # allow_nan=False for the same reason as the dashboard payload: bare
        # NaN and Infinity are not valid JSON and fail on the way back in.
        (d / "run.json").write_text(
            json.dumps(_json_safe(sidecar), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return d

    @classmethod
    def load(cls, directory: str | Path) -> BacktestResult:
        d = Path(directory)
        sidecar = json.loads((d / "run.json").read_text(encoding="utf-8"))
        folds = sidecar.get("folds")
        return cls(
            equity_curve=pl.read_parquet(d / "equity_curve.parquet"),
            positions=pl.read_parquet(d / "positions.parquet"),
            trades=pl.read_parquet(d / "trades.parquet"),
            metrics=sidecar["metrics"],
            run_meta=RunMeta.from_dict(sidecar["run_meta"]),
            folds=[Fold.from_dict(f) for f in folds] if folds else None,
        )

    def __repr__(self) -> str:
        n_folds = len(self.folds) if self.folds else 0
        return (
            f"BacktestResult(strategy={self.run_meta.strategy!r}, "
            f"days={len(self.equity_curve)}, trades={len(self.trades)}, "
            f"folds={n_folds})"
        )


def _json_safe(obj):
    """
    Minimal JSON coercion for the sidecar.

    Deliberately local rather than importing `export.encode.safe`: `export/` is
    the presentation layer and sits *above* `backtest/` in the dependency
    order, so importing it here would invert the arrow the architecture rests
    on.

    Branch order matters. np.bool_ is not a Python bool and np.int64 is not a
    Python int, so without explicit numpy branches they fall through to the
    final str() and a metric silently becomes the text "5" -- valid JSON,
    completely wrong, and invisible until something downstream tries to plot
    it. (np.float64 *does* subclass float, which is exactly why the asymmetry
    is easy to miss.)
    """
    import math

    import numpy as np

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)
