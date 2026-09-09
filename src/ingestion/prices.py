"""
Price ingestion — unified interface over multiple market-data sources.

Same design as `rates.py`, for the same reason: there is no "Yahoo ingester"
and "Binance ingester" as separate concerns visible to the rest of the system.
There are price SERIES, each of which knows which backend to pull from and what
calendar it trades on. Callers ask for a logical ticker and never care where it
came from.

Adding an asset class is therefore a registry entry plus a backend adapter —
never a change to analytics. If a new source ever requires touching code under
`analytics/`, the abstraction has failed and that is the finding to report.

Two details that matter in practice:

  * **The logical ticker is not the exchange symbol.** Binance calls it
    "BTC/USDT"; we call it "BTC-USD". The store writes one Parquet file per
    ticker, and a slash is not a legal filename — so the logical id is what
    reaches the store, and the exchange symbol stays inside the backend.

  * **BTC-USD is quoted in USDT, not USD.** The registry says so explicitly.
    Tether's peg has broken before (to ~$0.92 in October 2018), so a USDT pair
    is not a perfect USD series. It is the deepest, longest history available
    for free, which is the trade being made.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from src.domain.calendar import Calendar
from src.ingestion.base import BaseIngester


@dataclass(frozen=True)
class PriceSource:
    """Where a logical ticker comes from, and how its calendar behaves."""

    ticker: str            # our logical id, e.g. "BTC-USD"
    backend: str           # "yahoo" | "ccxt"
    code: str              # backend-native symbol, e.g. "BTC/USDT"
    calendar: Calendar
    asset_class: str
    exchange: str = ""     # CCXT venue, unused by Yahoo
    description: str = ""
    # Days between an observation's date and the moment it could be known.
    # A daily close is knowable at that close, so this is 0 for prices.
    #
    # Not to be confused with EXECUTION lag, which is a different thing and
    # belongs to the execution layer: knowing today's close does not mean you
    # could have traded at it. Conflating the two is a common way to smuggle
    # in a bar of look-ahead.
    publication_lag_days: int = 0


PRICE_REGISTRY: dict[str, PriceSource] = {
    "BTC-USD": PriceSource("BTC-USD", "ccxt", "BTC/USDT", Calendar.CONTINUOUS,
                           "crypto", "binance", "Bitcoin (Binance spot, USDT-quoted)"),
    "ETH-USD": PriceSource("ETH-USD", "ccxt", "ETH/USDT", Calendar.CONTINUOUS,
                           "crypto", "binance", "Ether (Binance spot, USDT-quoted)"),
    "SOL-USD": PriceSource("SOL-USD", "ccxt", "SOL/USDT", Calendar.CONTINUOUS,
                           "crypto", "binance", "Solana (Binance spot, USDT-quoted)"),
    "XRP-USD": PriceSource("XRP-USD", "ccxt", "XRP/USDT", Calendar.CONTINUOUS,
                           "crypto", "binance", "XRP (Binance spot, USDT-quoted)"),
}


def resolve(ticker: str) -> PriceSource:
    """
    Route a ticker to its source.

    Anything not explicitly registered falls through to Yahoo on the
    trading-day calendar. That fallback is deliberate: it keeps the universe
    open — any symbol Yahoo knows is usable without a code change — while
    letting assets whose calendar or venue differs declare themselves here.
    """
    if ticker in PRICE_REGISTRY:
        return PRICE_REGISTRY[ticker]
    return PriceSource(
        ticker=ticker, backend="yahoo", code=ticker,
        calendar=Calendar.TRADING_DAYS, asset_class="unknown",
    )


def publication_lag_days(ticker: str) -> int:
    """Days after which this ticker's observation for a date is knowable."""
    return resolve(ticker).publication_lag_days


def available_crypto() -> list[str]:
    """Registered crypto tickers."""
    return sorted(t for t, s in PRICE_REGISTRY.items() if s.asset_class == "crypto")


class PricesIngester(BaseIngester):
    """
    One ingester for every price source. Dispatches on the registry and
    relabels the result with the LOGICAL ticker, so downstream code (and the
    Parquet filenames) never see exchange-specific symbols.
    """

    source_name = "prices"

    def __init__(self) -> None:
        self._backends: dict[str, BaseIngester] = {}

    def _backend(self, source: PriceSource) -> BaseIngester:
        key = f"{source.backend}:{source.exchange}"
        if key not in self._backends:
            if source.backend == "yahoo":
                from src.ingestion.yahoo import YahooIngester
                self._backends[key] = YahooIngester()
            elif source.backend == "ccxt":
                from src.ingestion.crypto import CCXTIngester
                self._backends[key] = CCXTIngester(
                    exchange_id=source.exchange or "binance"
                )
            else:
                raise ValueError(
                    f"Unknown price backend {source.backend!r} for {source.ticker}"
                )
        return self._backends[key]

    def fetch(self, identifier: str, start: str,
              end: str | None = None) -> pl.DataFrame:
        source = resolve(identifier)
        backend = self._backend(source)
        if source.backend != "yahoo":
            logger.info(
                f"Price {identifier} -> {source.backend}"
                f"{':' + source.exchange if source.exchange else ''}:{source.code} "
                f"({source.calendar})"
            )
        df = backend.fetch(source.code, start, end)
        return df.with_columns(pl.lit(identifier).alias("ticker"))

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("adj_close").is_not_null() & (pl.col("adj_close") > 0))
