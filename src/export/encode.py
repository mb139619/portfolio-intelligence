"""
JSON encoding for the dashboard payload.

Two jobs, both boring and both easy to get subtly wrong:

1. Convert numpy/polars/date values into plain Python. The analytics return
   np.float64, np.ndarray and pl.DataFrame everywhere; json.dumps rejects all
   of them.

2. Replace NaN and infinity with null. This one matters more than it looks:
   Python's json.dumps happily emits bare `NaN` and `Infinity` tokens, which
   are NOT valid JSON, and the browser's JSON.parse throws on them. The values
   arise naturally here — a regime with one observation has NaN volatility, and
   EVT expected shortfall is infinite when the GPD shape parameter ξ ≥ 1. So
   the sanitising is load-bearing, not defensive decoration.

The section/stat/table helpers below define the payload schema the SPA renders.
Keeping that schema in one file means a new analytic is a new section in Python,
with no JavaScript change at all.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl

# ──────────────────────────────────────────────────────────────────────────
# Value conversion
# ──────────────────────────────────────────────────────────────────────────

def safe(obj: Any) -> Any:
    """
    Recursively convert `obj` into something json.dumps can emit as valid JSON.
    Non-finite floats become None so the browser sees `null` rather than a
    parse error.
    """
    # Order matters: np.bool_ is not a Python bool, and bool is a subclass of
    # int, so booleans must be tested before the numeric branches.
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [safe(v) for v in obj.tolist()]
    if isinstance(obj, pl.DataFrame):
        return [safe(row) for row in obj.to_dicts()]
    if isinstance(obj, pl.Series):
        return [safe(v) for v in obj.to_list()]
    if isinstance(obj, dict):
        return {str(k): safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe(v) for v in obj]
    return str(obj)


def figure(fig, fid: str, title: str = "", note: str = "") -> dict:
    """
    Serialise a Plotly figure to the payload shape.

    plotly.io.to_json is used rather than fig.to_dict() because it already
    encodes numpy arrays and maps NaN to null; round-tripping through json.loads
    then gives plain Python that our own encoder never has to touch again.
    """
    import plotly.io as pio

    spec = json.loads(pio.to_json(fig, validate=False))
    layout = spec.get("layout", {})

    # The card renders `title` in its own header, so leaving the figure's
    # internal title in place would print the same words twice, one above the
    # other, on every chart. The figures keep their titles for notebook use;
    # the dashboard just stops drawing them.
    if title:
        layout.pop("title", None)

    return {
        "id": fid,
        "title": title,
        "note": note,
        "data": spec.get("data", []),
        "layout": layout,
    }


# ──────────────────────────────────────────────────────────────────────────
# Payload schema
# ──────────────────────────────────────────────────────────────────────────

def stat(
    label: str,
    value: Any,
    fmt: str = "number",
    hint: str = "",
    tone: str = "neutral",
) -> dict:
    """
    One KPI tile.

    fmt  : "percent" | "number" | "ratio" | "integer" | "text" — how the SPA
           formats the raw value. Formatting is deliberately the frontend's job:
           the payload carries numbers, not pre-rendered strings, so the same
           data stays usable by anything else that reads the JSON.
    tone : "neutral" | "good" | "bad" — optional colour emphasis.
    """
    return {
        "label": label,
        "value": safe(value),
        "format": fmt,
        "hint": hint,
        "tone": tone,
    }


def table(
    tid: str,
    title: str,
    rows: pl.DataFrame | list[dict],
    formats: dict[str, str] | None = None,
    note: str = "",
    columns: list[str] | None = None,
) -> dict:
    """
    A tabular block. `formats` maps column name -> format token (same
    vocabulary as `stat`), so the SPA can right-align and render percentages
    without knowing what the column means.
    """
    data = safe(rows)
    if columns is None:
        columns = list(data[0].keys()) if data else []
    return {
        "id": tid,
        "title": title,
        "columns": columns,
        "rows": data,
        "formats": formats or {},
        "note": note,
    }


def section(
    sid: str,
    title: str,
    subtitle: str = "",
    stats: Iterable[dict] | None = None,
    figures: Iterable[dict] | None = None,
    tables: Iterable[dict] | None = None,
    notes: Iterable[str] | None = None,
    error: str = "",
) -> dict:
    """
    One page of the dashboard. `error` is set when the analytic could not run
    (too little history, missing factor data); the SPA renders the message in
    place of the content rather than silently dropping the page, because a
    quietly missing risk panel is worse than a visibly broken one.
    """
    return {
        "id": sid,
        "title": title,
        "subtitle": subtitle,
        "stats": list(stats or []),
        "figures": list(figures or []),
        "tables": list(tables or []),
        "notes": list(notes or []),
        "error": error,
    }


def dumps(payload: dict, indent: int | None = None) -> str:
    """
    Serialise the payload. allow_nan=False turns any non-finite value that
    slipped past `safe` into a loud error at build time instead of a page that
    silently fails to load in the browser.
    """
    return json.dumps(payload, indent=indent, allow_nan=False, ensure_ascii=False)
