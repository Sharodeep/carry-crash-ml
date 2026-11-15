"""FRED history download for the crash early-warning research pipeline.

Raw series are cached to research/cache/{series_id}.csv and only
re-downloaded when refresh=True. This module hands back already-normalized
series (FX in USD-per-foreign terms, rates spliced where the primary OECD
series has gone stale) -- weekly resampling, lagging, and basket
construction are build.py's job, not this module's.
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from config import (
    RESEARCH_CACHE_DIR,
    RESEARCH_FX_SERIES,
    RESEARCH_RATE_SERIES,
    RESEARCH_RISK_SERIES,
    RESEARCH_STALE_SPLICE_MONTHS,
)

load_dotenv()


def get_fred_client() -> Fred:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Add it to .env or export it in your shell."
        )
    return Fred(api_key=api_key)


def _cache_path(series_id: str) -> str:
    os.makedirs(RESEARCH_CACHE_DIR, exist_ok=True)
    return os.path.join(RESEARCH_CACHE_DIR, f"{series_id}.csv")


def fetch_raw_series(fred: Fred, series_id: str, refresh: bool = False) -> pd.Series:
    """Fetch one FRED series, cached as CSV. Index is a DatetimeIndex."""
    path = _cache_path(series_id)
    if not refresh and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.iloc[:, 0].dropna()

    series = fred.get_series(series_id).dropna()
    series.to_frame(name=series_id).to_csv(path)
    return series


def fetch_all_fx(fred: Fred, refresh: bool = False) -> dict:
    """USD per 1 unit of foreign currency, for every currency in RESEARCH_FX_SERIES."""
    fx = {}
    for currency, info in RESEARCH_FX_SERIES.items():
        series = fetch_raw_series(fred, info["series_id"], refresh=refresh)
        if info["invert"]:
            series = 1.0 / series
        fx[currency] = series
    return fx


def fetch_all_rates(fred: Fred, refresh: bool = False) -> tuple[dict, list[str]]:
    """3-month-proxy policy rates, gap-filled from the splice series where
    the primary series is missing data at either end (stale tail, or a
    head gap like JPY's IR3TIB01 only starting in 2002).

    Returns (dict[currency] -> pd.Series, list of log messages describing
    any gap-fills or special-cased series that were applied).
    """
    rates = {}
    log = []

    for currency, info in RESEARCH_RATE_SERIES.items():
        primary = fetch_raw_series(fred, info["primary"], refresh=refresh)

        if currency == "EUR":
            monthly = primary.resample("MS").last()
            rates[currency] = monthly
            log.append(
                f"EUR: using {info['primary']} (daily, resampled to monthly) "
                "-- no live OECD 3-month series available for the Eurozone."
            )
            continue

        if info["splice"] is None:
            rates[currency] = primary
            continue

        last_date = primary.index.max()
        months_stale = (pd.Timestamp.today() - last_date).days / 30.44
        if months_stale <= RESEARCH_STALE_SPLICE_MONTHS and primary.index.min() <= pd.Timestamp("2000-06-01"):
            rates[currency] = primary
            continue

        splice_series = fetch_raw_series(fred, info["splice"], refresh=refresh)
        combined = primary.combine_first(splice_series).dropna()
        filled_dates = combined.index.difference(primary.index)
        rates[currency] = combined
        if len(filled_dates):
            head_fill = (filled_dates < primary.index.min()).sum()
            tail_fill = (filled_dates > primary.index.max()).sum()
            log.append(
                f"{currency}: {info['primary']} gap-filled from {info['splice']} "
                f"({head_fill} head observation(s) before {primary.index.min().date()}, "
                f"{tail_fill} tail observation(s) after {primary.index.max().date()})."
            )

    return rates, log


def fetch_all_risk(fred: Fred, refresh: bool = False) -> dict:
    return {
        name: fetch_raw_series(fred, series_id, refresh=refresh)
        for name, series_id in RESEARCH_RISK_SERIES.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache research FRED series.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download.")
    args = parser.parse_args()

    fred = get_fred_client()

    fx = fetch_all_fx(fred, refresh=args.refresh)
    rates, log = fetch_all_rates(fred, refresh=args.refresh)
    risk = fetch_all_risk(fred, refresh=args.refresh)

    print(f"FX series ({len(fx)}):")
    for currency, series in fx.items():
        print(f"  {currency}: {series.index.min().date()} -> {series.index.max().date()} (n={len(series)})")

    print(f"\nRate series ({len(rates)}):")
    for currency, series in rates.items():
        print(f"  {currency}: {series.index.min().date()} -> {series.index.max().date()} (n={len(series)})")

    print(f"\nRisk series ({len(risk)}):")
    for name, series in risk.items():
        print(f"  {name}: {series.index.min().date()} -> {series.index.max().date()} (n={len(series)})")

    if log:
        print("\nSplices / special cases:")
        for line in log:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
