"""
Crypto OHLCV via CCXT.

CCXT normalises a hundred exchanges behind one interface, so this module is a
thin adapter: pagination, unit conversion, and the schema the store expects.
Nothing here knows about portfolios or analytics.

Two honest notes about the data, both of which belong in METHODOLOGY.md rather
than being silently smoothed over:

  * **adj_close == close.** Equity adjusted closes fold in dividends and splits.
    A spot crypto pair has neither, so the column exists only to keep one schema
    across sources — it is not an adjustment, and pretending otherwise would
    imply a correction that was never applied.

  * **Bar close misalignment.** Binance daily candles close at 00:00 UTC; US
    equities close at 16:00 ET (21:00/20:00 UTC). A daily crypto bar therefore
    ends 3-4 hours *after* the equity bar it is dated alongside and contains
    several hours of news the equity bar could not. This induces spurious
    lead-lag structure in cross-asset correlations. We keep exchange-native
    dating and state the convention; re-basing crypto to the equity close would
    trade a documented bias for an undocumented one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
from loguru import logger

from src.ingestion.base import BaseIngester

# Binance returns at most 1000 candles per call.
_PAGE_LIMIT = 1000
_DAY_MS = 86_400_000


def _to_millis(day: str | date) -> int:
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


class CCXTIngester(BaseIngester):
    """
    Daily OHLCV from any CCXT-supported exchange. Defaults to Binance spot.

    The identifier is the exchange-native symbol (e.g. "BTC/USDT"); the mapping
    from our logical ticker to that symbol lives in the price registry, not here.
    """

    source_name = "ccxt"

    def __init__(self, exchange_id: str = "binance", timeframe: str = "1d") -> None:
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self._exchange = None

    @property
    def exchange(self):
        # Built lazily so importing this module never opens a network client —
        # the test suite imports the ingestion package without touching Binance.
        if self._exchange is None:
            import ccxt

            if not hasattr(ccxt, self.exchange_id):
                raise ValueError(
                    f"Unknown CCXT exchange {self.exchange_id!r}. "
                    f"Try one of: binance, coinbase, kraken, bitstamp."
                )
            self._exchange = getattr(ccxt, self.exchange_id)({
                "enableRateLimit": True,   # CCXT throttles itself to the venue's limit
                "timeout": 30_000,
            })
        return self._exchange

    def fetch(self, identifier: str, start: str,
              end: str | None = None) -> pl.DataFrame:
        since = _to_millis(start)
        end_ms = _to_millis(end) if end else None

        rows: list[list] = []
        seen_last: int | None = None

        while True:
            batch = self.exchange.fetch_ohlcv(
                identifier, timeframe=self.timeframe, since=since, limit=_PAGE_LIMIT
            )
            if not batch:
                break

            if end_ms is not None:
                batch = [c for c in batch if c[0] <= end_ms]
                if not batch:
                    break

            rows.extend(batch)
            last = batch[-1][0]

            # Guard against a venue that ignores `since` and returns the same
            # page forever — without this the loop would never terminate.
            if seen_last is not None and last <= seen_last:
                break
            seen_last = last

            if len(batch) < _PAGE_LIMIT:
                break
            since = last + _DAY_MS

        if not rows:
            raise ValueError(
                f"No {self.timeframe} data returned for {identifier} on "
                f"{self.exchange_id} from {start}"
            )

        logger.debug(f"{self.exchange_id}/{identifier}: {len(rows)} candles")

        df = pl.DataFrame(
            rows,
            schema=["ts", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        return (
            df.with_columns([
                pl.from_epoch(pl.col("ts"), time_unit="ms").dt.date().alias("date"),
                pl.lit(identifier).alias("ticker"),
                # Not an adjustment — see the module docstring.
                pl.col("close").alias("adj_close"),
            ])
            .select(["date", "ticker", "open", "high", "low", "close",
                     "adj_close", "volume"])
            .unique(subset=["date"], keep="last")
            .sort("date")
        )

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("adj_close").is_not_null() & (pl.col("adj_close") > 0))
