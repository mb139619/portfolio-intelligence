"""
Stress scenario definitions.

Two kinds of stress:
  - HISTORICAL: replay a real crisis window. Either on the actual asset returns
    (if the store has data for that era) or — more robustly — by applying the
    portfolio's current factor betas to the factor returns of that window
    (Fama-French factors reach back to 1926, so this works for any crisis).
  - PARAMETRIC: apply a hypothetical shock. On equity-style factors directly,
    or on macro variables (rates, oil, USD, VIX) via estimated sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalScenario:
    name: str
    start: str
    end: str
    description: str


# Canonical crisis windows (peak-to-trough-ish).
HISTORICAL_SCENARIOS: dict[str, HistoricalScenario] = {
    "GFC_2008": HistoricalScenario(
        "GFC_2008", "2008-09-01", "2009-03-09",
        "Global Financial Crisis: Lehman collapse to market trough",
    ),
    "EU_CRISIS_2011": HistoricalScenario(
        "EU_CRISIS_2011", "2011-07-01", "2011-10-03",
        "Euro sovereign debt crisis / US downgrade selloff",
    ),
    "TAPER_2013": HistoricalScenario(
        "TAPER_2013", "2013-05-22", "2013-06-24",
        "Taper tantrum: rates spike on Fed tapering signal",
    ),
    "VOLMAGEDDON_2018": HistoricalScenario(
        "VOLMAGEDDON_2018", "2018-10-01", "2018-12-24",
        "Q4 2018 selloff: rate-hike fears, growth scare",
    ),
    "COVID_2020": HistoricalScenario(
        "COVID_2020", "2020-02-19", "2020-03-23",
        "COVID-19 crash: fastest bear market on record",
    ),
    "INFLATION_2022": HistoricalScenario(
        "INFLATION_2022", "2022-01-01", "2022-10-14",
        "2022 inflation shock: aggressive Fed hikes, stocks and bonds down together",
    ),
}


@dataclass(frozen=True)
class FactorShock:
    """A hypothetical move applied to factor returns, e.g. {'Mkt-RF': -0.15}."""
    name: str
    moves: dict  # factor_name -> shock (decimal return)
    description: str = ""


# A few ready-made factor shocks (FF5 space).
FACTOR_SHOCKS: dict[str, FactorShock] = {
    "EQUITY_CRASH": FactorShock(
        "EQUITY_CRASH", {"Mkt-RF": -0.20},
        "Broad equity market down 20%",
    ),
    "VALUE_ROTATION": FactorShock(
        "VALUE_ROTATION", {"HML": 0.10, "Mkt-RF": -0.05},
        "Sharp rotation into value, mild market decline",
    ),
    "QUALITY_FLIGHT": FactorShock(
        "QUALITY_FLIGHT", {"RMW": 0.08, "Mkt-RF": -0.10},
        "Flight to quality during a market drop",
    ),
}


@dataclass(frozen=True)
class MacroShock:
    """
    A hypothetical move on macro variables, in their NATIVE units:
      - rate variables: change in yield as a decimal (0.01 = +100bp)
      - return variables (oil, USD): decimal return (-0.30 = -30%)
      - level variables (VIX): change in level (+20 = +20 vol points)
    e.g. {'rate_10y': 0.01} for a +100bp rate shock.
    """
    name: str
    moves: dict
    description: str = ""


MACRO_SHOCKS: dict[str, MacroShock] = {
    "RATES_UP_100": MacroShock("RATES_UP_100", {"rate_10y": 0.01}, "+100bp on the 10Y"),
    "RATES_UP_200": MacroShock("RATES_UP_200", {"rate_10y": 0.02}, "+200bp on the 10Y"),
    "OIL_SHOCK": MacroShock("OIL_SHOCK", {"oil": 0.40}, "+40% oil price spike"),
    "USD_SHOCK": MacroShock("USD_SHOCK", {"usd": 0.10}, "+10% USD appreciation"),
    "VIX_SHOCK": MacroShock("VIX_SHOCK", {"vix": 20.0}, "+20 point VIX spike"),
}
