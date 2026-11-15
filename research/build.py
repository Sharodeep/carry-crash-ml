"""Weekly panel, excess returns, and the G10 carry basket.

No look-ahead: rates are shifted for publication lag before being
forward-filled to weekly, and both the ranking and the excess-return
formula use the rate known as of the END of week t-1 (one more weekly
shift beyond the publication lag).
"""

import numpy as np
import pandas as pd

from config import (
    RESEARCH_N_LONG,
    RESEARCH_N_SHORT,
    RESEARCH_PUBLICATION_LAG_MONTHS,
    RESEARCH_SAMPLE_START,
    RESEARCH_WEEKLY_RULE,
)
from research.fetch import fetch_all_fx, fetch_all_rates, fetch_all_risk, get_fred_client

ALL_CURRENCIES = ["USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "NOK", "SEK"]
NON_USD_CURRENCIES = [c for c in ALL_CURRENCIES if c != "USD"]


def _weekly_index(start: str, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=RESEARCH_WEEKLY_RULE)


def _rate_to_weekly(rate: pd.Series, weekly_index: pd.DatetimeIndex) -> pd.Series:
    """Publication-lag shift, then forward-fill onto the weekly grid."""
    lagged = rate.copy()
    lagged.index = lagged.index + pd.DateOffset(months=RESEARCH_PUBLICATION_LAG_MONTHS)
    lagged = lagged.sort_index()
    union_index = lagged.index.union(weekly_index)
    filled = lagged.reindex(union_index).ffill()
    return filled.reindex(weekly_index)


def build_weekly_panel(refresh: bool = False) -> dict:
    """Fetch, lag, and align everything onto one weekly (W-FRI) grid.

    Returns a dict with: fx_weekly (DataFrame, USD per foreign, non-USD
    currencies), rate_weekly (DataFrame, percent, all 10 currencies),
    risk_weekly (DataFrame: vix, baa10y, brent), and fetch_log.
    """
    fred = get_fred_client()
    fx_raw = fetch_all_fx(fred, refresh=refresh)
    rates_raw, fetch_log = fetch_all_rates(fred, refresh=refresh)
    risk_raw = fetch_all_risk(fred, refresh=refresh)

    # Only FX bounds the weekly grid's end date. Rates and risk series are
    # both forward-filled onto whatever weekly_index they're given
    # (_rate_to_weekly, and risk_weekly's own .reindex().ffill()), which is
    # exactly the staleness-but-not-missing behaviour current_reading.py's
    # stale_inputs check is meant to flag -- there's no technical reason to
    # truncate the whole panel by whichever single raw series (e.g. a
    # slow-to-update commodity price) happens to be most stale that week.
    end = min(s.index.max() for s in fx_raw.values())
    weekly_index = _weekly_index(RESEARCH_SAMPLE_START, end)

    fx_weekly = pd.DataFrame(
        {c: fx_raw[c].resample(RESEARCH_WEEKLY_RULE).last() for c in NON_USD_CURRENCIES}
    ).reindex(weekly_index).ffill()

    rate_weekly = pd.DataFrame(
        {c: _rate_to_weekly(rates_raw[c], weekly_index) for c in ALL_CURRENCIES}
    )

    risk_weekly = pd.DataFrame(
        {name: s.resample(RESEARCH_WEEKLY_RULE).last() for name, s in risk_raw.items()}
    ).reindex(weekly_index).ffill()

    return {
        "fx_weekly": fx_weekly,
        "rate_weekly": rate_weekly,
        "risk_weekly": risk_weekly,
        "fetch_log": fetch_log,
    }


def compute_excess_returns(fx_weekly: pd.DataFrame, rate_weekly: pd.DataFrame) -> pd.DataFrame:
    """Weekly log excess return vs USD for every currency (USD's own xr = 0)."""
    log_ret = np.log(fx_weekly / fx_weekly.shift(1))
    rate_decimal_lag1 = rate_weekly.shift(1) / 100.0

    xr = pd.DataFrame(index=fx_weekly.index)
    xr["USD"] = 0.0
    for currency in NON_USD_CURRENCIES:
        xr[currency] = log_ret[currency] + (
            rate_decimal_lag1[currency] - rate_decimal_lag1["USD"]
        ) / 52.0

    return xr[ALL_CURRENCIES]


def build_carry_basket(xr: pd.DataFrame, rate_weekly: pd.DataFrame) -> pd.DataFrame:
    """Rank by lagged rate each week; long top N_LONG, short bottom N_SHORT."""
    rate_rank_basis = rate_weekly.shift(1)[ALL_CURRENCIES]

    rows = []
    for t in xr.index:
        ranks = rate_rank_basis.loc[t]
        if ranks.isna().any() or xr.loc[t, ALL_CURRENCIES].isna().any():
            rows.append({"date": t, "b_t": np.nan, "longs": None, "shorts": None})
            continue

        ordered = ranks.sort_values(ascending=False)
        longs = list(ordered.index[:RESEARCH_N_LONG])
        shorts = list(ordered.index[-RESEARCH_N_SHORT:])

        b_t = xr.loc[t, longs].mean() - xr.loc[t, shorts].mean()
        rows.append({"date": t, "b_t": b_t, "longs": longs, "shorts": shorts})

    basket = pd.DataFrame(rows).set_index("date")
    return basket.dropna(subset=["b_t"])


def run_sanity_checks(basket: pd.DataFrame) -> str:
    lines = []

    cumulative = basket["b_t"].cumsum()
    lines.append(f"Sample: {basket.index.min().date()} -> {basket.index.max().date()} ({len(basket)} weeks)")
    lines.append(f"Cumulative basket return (log, sum): {cumulative.iloc[-1]:.3f}")

    episodes = {
        "2008 GFC (Sep-Dec)": ("2008-09-01", "2008-12-31"),
        "Aug 2011": ("2011-08-01", "2011-08-31"),
        "Aug 2015": ("2015-08-01", "2015-08-31"),
        "Mar 2020": ("2020-03-01", "2020-03-31"),
        "Jul-Aug 2024": ("2024-07-01", "2024-08-31"),
    }
    lines.append("\nEpisode returns (sum of weekly b_t):")
    for name, (start, end) in episodes.items():
        window = basket.loc[start:end, "b_t"]
        if window.empty:
            lines.append(f"  {name}: no data in window")
        else:
            lines.append(f"  {name}: {window.sum():+.4f} over {len(window)} weeks")

    long_counts = pd.Series(
        [c for longs in basket["longs"] for c in longs]
    ).value_counts()
    short_counts = pd.Series(
        [c for shorts in basket["shorts"] for c in shorts]
    ).value_counts()
    lines.append("\nWeeks spent in long leg:")
    lines.append(long_counts.reindex(ALL_CURRENCIES, fill_value=0).to_string())
    lines.append("\nWeeks spent in short leg:")
    lines.append(short_counts.reindex(ALL_CURRENCIES, fill_value=0).to_string())

    mean_weekly = basket["b_t"].mean()
    std_weekly = basket["b_t"].std()
    sharpe = mean_weekly / std_weekly * np.sqrt(52)
    lines.append(f"\nFull-sample annualised Sharpe: {sharpe:.3f}")
    if sharpe > 1.5 or sharpe < -0.5:
        lines.append(
            "  WARNING: Sharpe outside the expected 0 to 0.8 range -- check FX "
            "inversion and percent-vs-decimal units before proceeding."
        )

    return "\n".join(lines)


def main(refresh: bool = False):
    import os

    panel = build_weekly_panel(refresh=refresh)
    if panel["fetch_log"]:
        print("Fetch log:")
        for line in panel["fetch_log"]:
            print(f"  - {line}")
        print()

    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])

    os.makedirs("outputs/tables", exist_ok=True)
    basket.to_csv("outputs/tables/basket_weekly.csv")

    print(run_sanity_checks(basket))
    return panel, xr, basket


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    main(refresh=args.refresh)
