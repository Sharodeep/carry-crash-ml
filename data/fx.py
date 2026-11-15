"""Spot FX fetching and rolling volatility via yfinance.

Every currency is fetched once against USD (a single batched yfinance call),
and cross pairs are derived by triangulation: LONG/SHORT = (LONG/USD) /
(SHORT/USD). This is exact, not approximate -- log-return(LONG/SHORT) ==
log-return(LONG/USD) - log-return(SHORT/USD) by construction -- and it's how
the market itself prices illiquid crosses that have no direct quote. It also
turns what would be ~110 pair-by-pair downloads (many for tickers that don't
exist on Yahoo) into a single batched request.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import (
    DRIFT_WINDOW_DAYS,
    FX_HISTORY_DAYS,
    MANAGED_CURRENCY_VOL_THRESHOLD_PCT,
    TRADING_DAYS_PER_YEAR,
    VOL_ROLLING_WINDOW,
    VOL_ROLLING_WINDOW_EM,
    VOL_ROLLING_WINDOW_LONG,
)


@st.cache_data(ttl=900)
def fetch_usd_quotes(currencies: tuple[str, ...]) -> tuple[dict, list[str]]:
    """Fetch each currency's price in USD terms (CCY/USD), batched.

    Returns (dict[currency] -> pd.Series, warnings) for currencies whose
    Yahoo ticker (`{CCY}USD=X`) returned data.
    """
    non_usd = [c for c in currencies if c != "USD"]
    if not non_usd:
        return {}, []

    tickers = [f"{c}USD=X" for c in non_usd]
    data = yf.download(
        tickers,
        period=f"{FX_HISTORY_DAYS}d",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    quotes = {}
    warnings = []
    available_tickers = (
        set(data.columns.get_level_values(0)) if not data.empty else set()
    )
    for currency in non_usd:
        ticker = f"{currency}USD=X"
        if ticker not in available_tickers:
            warnings.append(f"{currency}: no yfinance data for {ticker}.")
            continue
        close = data[ticker]["Close"].dropna()
        if close.empty:
            warnings.append(f"{currency}: no yfinance data for {ticker}.")
            continue
        quotes[currency] = close

    return quotes, warnings


def _pair_series(long_ccy: str, short_ccy: str, usd_quotes: dict) -> pd.Series | None:
    if long_ccy == short_ccy:
        return None

    if long_ccy == "USD":
        short_series = usd_quotes.get(short_ccy)
        if short_series is None:
            return None
        return 1.0 / short_series

    if short_ccy == "USD":
        return usd_quotes.get(long_ccy)

    long_series = usd_quotes.get(long_ccy)
    short_series = usd_quotes.get(short_ccy)
    if long_series is None or short_series is None:
        return None

    aligned = pd.concat(
        [long_series.rename("long"), short_series.rename("short")], axis=1
    ).dropna()
    if aligned.empty:
        return None
    return aligned["long"] / aligned["short"]


def compute_annualized_vol(
    close_series: pd.Series, window: int = VOL_ROLLING_WINDOW
) -> float | None:
    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    if len(log_returns) < window:
        return None
    rolling_vol = log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_vol = rolling_vol.dropna()
    if rolling_vol.empty:
        return None
    return float(rolling_vol.iloc[-1]) * 100


def compute_trailing_drift_pct(
    close_series: pd.Series, window_days: int = DRIFT_WINDOW_DAYS
) -> float | None:
    """Raw (not annualised) cumulative trend of the pair over a trailing
    calendar window -- the window itself is 12 months, so there's no separate
    annualisation step.

    Positive = the long currency appreciated vs. the short currency over the
    window; negative = it depreciated. Feeds the carry adjustment, which only
    ever subtracts depreciation -- appreciation must never add to carry.
    """
    if close_series.empty:
        return None
    cutoff = close_series.index[-1] - pd.Timedelta(days=window_days)
    window_series = close_series[close_series.index >= cutoff]
    if len(window_series) < 20:
        return None
    start_price, end_price = window_series.iloc[0], window_series.iloc[-1]
    if start_price <= 0:
        return None
    return float(np.log(end_price / start_price)) * 100


def fetch_currency_managed_flags(currencies: tuple[str, ...]) -> dict:
    """Per-currency 'managed' flag: 1-year vol (vs. USD) below the threshold.

    A currency whose outright vol vs. USD is this low likely moves on a
    steady, managed path whose day-to-day volatility understates the risk of
    a step devaluation. Returns dict[currency] -> bool.
    """
    usd_quotes, _ = fetch_usd_quotes(currencies)
    flags = {"USD": False}
    for currency in currencies:
        if currency == "USD":
            continue
        series = usd_quotes.get(currency)
        vol_1y = compute_annualized_vol(series, window=VOL_ROLLING_WINDOW_LONG) if series is not None else None
        flags[currency] = vol_1y is not None and vol_1y < MANAGED_CURRENCY_VOL_THRESHOLD_PCT
    return flags


def fetch_all_pair_metrics(
    pairs: list[tuple[str, str]]
) -> tuple[dict, list[str]]:
    """Per-pair vol (30d, ~1y, ~5y) and trailing drift, via USD triangulation.

    Returns (dict[(long, short)] -> {"vol_30d_pct", "vol_1y_pct", "vol_5y_pct",
    "vol_g10_pct" (max of 30d/1y -- what the G10 tab ranks on), "drift_pct"},
    warnings). The EM tab ranks on vol_5y_pct directly, so a calm recent
    half-decade can't hide an older step-devaluation.
    """
    currencies = tuple(sorted({c for pair in pairs for c in pair}))
    usd_quotes, warnings = fetch_usd_quotes(currencies)

    metrics = {}
    for long_ccy, short_ccy in pairs:
        series = _pair_series(long_ccy, short_ccy, usd_quotes)
        if series is None:
            continue
        vol_30d = compute_annualized_vol(series, window=VOL_ROLLING_WINDOW)
        vol_1y = compute_annualized_vol(series, window=VOL_ROLLING_WINDOW_LONG)
        vol_5y = compute_annualized_vol(series, window=VOL_ROLLING_WINDOW_EM)
        g10_candidates = [v for v in (vol_30d, vol_1y) if v is not None]
        if not g10_candidates and vol_5y is None:
            continue
        drift_pct = compute_trailing_drift_pct(series)
        metrics[(long_ccy, short_ccy)] = {
            "vol_30d_pct": vol_30d,
            "vol_1y_pct": vol_1y,
            "vol_5y_pct": vol_5y,
            "vol_g10_pct": max(g10_candidates) if g10_candidates else None,
            "drift_pct": drift_pct if drift_pct is not None else 0.0,
        }

    missing_pairs = [
        f"{long_ccy}/{short_ccy}"
        for long_ccy, short_ccy in pairs
        if (long_ccy, short_ccy) not in metrics
    ]
    if missing_pairs:
        warnings.append(
            f"{len(missing_pairs)} pair(s) excluded from ranking due to "
            "insufficient FX data: " + ", ".join(missing_pairs)
        )

    return metrics, warnings
