"""FRED policy rate fetching and EM static fallback rates."""

import os
from datetime import date

import pandas as pd
import streamlit as st
from fredapi import Fred

from config import EM_FALLBACK_RATES, FRED_SERIES

_PRIOR_RATES_KEY = "_prior_fred_rates"


class FredKeyMissingError(RuntimeError):
    """Raised when no FRED API key is configured."""


def get_fred_client() -> Fred:
    try:
        api_key = st.secrets["fred"]["api_key"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise FredKeyMissingError(
            "FRED API key not found. Add it to .streamlit/secrets.toml or set "
            "the FRED_API_KEY environment variable."
        )
    return Fred(api_key=api_key)


def _fetch_latest_observation(fred: Fred, series_id: str):
    series = fred.get_series(series_id).dropna()
    if series.empty:
        return None, None
    last_date = series.index[-1]
    last_value = float(series.iloc[-1])
    return last_value, last_date.strftime("%Y-%m-%d")


@st.cache_data(ttl=3600)
def _fetch_all_fred_rates(_fred: Fred) -> dict:
    """Fetch the latest observation for every FRED-backed currency.

    Keyed by currency, values are (rate, last_updated) or None on failure.
    """
    results = {}
    for currency, series_id in FRED_SERIES.items():
        try:
            results[currency] = _fetch_latest_observation(_fred, series_id)
        except Exception:
            results[currency] = (None, None)
    return results


def build_rate_table(fred: Fred) -> tuple[pd.DataFrame, list[str]]:
    """Build the rate table for all currencies.

    Returns (DataFrame[currency, rate, source, last_updated], warnings).
    """
    warnings: list[str] = []
    rows = []

    prior_rates = st.session_state.setdefault(_PRIOR_RATES_KEY, {})
    fred_results = _fetch_all_fred_rates(fred)

    for currency, (rate, last_updated) in fred_results.items():
        source = "FRED"
        if rate is None:
            if currency in prior_rates:
                rate, last_updated = prior_rates[currency]
                warnings.append(
                    f"{currency}: FRED returned no data, using last cached rate "
                    f"({last_updated})."
                )
            elif currency in EM_FALLBACK_RATES:
                fallback = EM_FALLBACK_RATES[currency]
                rate, last_updated = fallback["rate"], fallback["last_updated"]
                source = "Static"
                warnings.append(
                    f"{currency}: FRED returned no data and no cached rate is "
                    f"available, using emergency static rate ({last_updated})."
                )
            else:
                warnings.append(
                    f"{currency}: FRED returned no data and no cached or "
                    "static rate is available. Excluded from screener."
                )
                continue
        else:
            prior_rates[currency] = (rate, last_updated)

        rows.append(
            {
                "currency": currency,
                "rate": rate,
                "source": source,
                "last_updated": last_updated,
            }
        )

    return pd.DataFrame(rows), warnings


def em_fallback_is_stale(last_updated: str, stale_days: int) -> bool:
    """last_updated is a 'YYYY-MM' string; compare against today."""
    updated_date = date.fromisoformat(f"{last_updated}-01")
    return (date.today() - updated_date).days > stale_days
