"""Carry computation, vol overlay, and ranking logic."""

from itertools import permutations

import pandas as pd

from config import SIGNAL_MODERATE_THRESHOLD, SIGNAL_STRONG_THRESHOLD
from data.fx import fetch_all_pair_metrics, fetch_currency_managed_flags


def build_carry_table(rate_df: pd.DataFrame, managed_flags: dict) -> pd.DataFrame:
    """Every ordered long/short currency combination and its carry."""
    rates = dict(zip(rate_df["currency"], rate_df["rate"]))
    sources = dict(zip(rate_df["currency"], rate_df["source"]))
    rows = []
    for long_ccy, short_ccy in permutations(rates, 2):
        long_rate = rates[long_ccy]
        short_rate = rates[short_ccy]
        rows.append(
            {
                "pair": f"{long_ccy}/{short_ccy}",
                "long_ccy": long_ccy,
                "short_ccy": short_ccy,
                "long_rate": long_rate,
                "short_rate": short_rate,
                "carry_pct": long_rate - short_rate,
                "stale_rate": sources[long_ccy] == "Static" or sources[short_ccy] == "Static",
                "managed_rate": managed_flags.get(long_ccy, False) or managed_flags.get(short_ccy, False),
            }
        )
    return pd.DataFrame(rows)


def _signal_for(carry_to_vol: float) -> str:
    if carry_to_vol < 0 or carry_to_vol < SIGNAL_MODERATE_THRESHOLD:
        return "Weak / Avoid"
    if carry_to_vol > SIGNAL_STRONG_THRESHOLD:
        return "Strong"
    return "Moderate"


def build_screener_table(rate_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Full pipeline: carry -> drift-adjusted carry -> vol overlay -> rank -> signal.

    Two different ranking metrics, one per tab:
    - G10 ranks on raw carry_pct / vol_g10_pct (vol_g10_pct = max of 30-day
      and ~1-year annualised vol). adj_carry_pct is shown as an info column
      only -- it is not used to rank or signal G10 pairs.
    - EM ranks on adj_carry_pct / vol_5y_pct (a ~5-year annualised vol, so a
      calm recent half-decade can't hide an older step-devaluation).
      adj_carry_pct = carry_pct - max(trailing 12-month depreciation, 0): it
      only ever subtracts depreciation of the long currency vs. the short
      currency -- appreciation never adds to carry.

    Returns (DataFrame, warnings). DataFrame columns:
    [rank, pair, long_ccy, short_ccy, long_rate, short_rate, carry_pct,
     drift_pct, adj_carry_pct, vol_30d_pct, vol_1y_pct, vol_5y_pct,
     vol_g10_pct, carry_to_vol_g10, adj_carry_to_vol_em, signal_g10,
     signal_em, stale_rate, managed_rate]
    """
    managed_flags = fetch_currency_managed_flags(tuple(rate_df["currency"]))
    carry_df = build_carry_table(rate_df, managed_flags)

    pairs = list(zip(carry_df["long_ccy"], carry_df["short_ccy"]))
    metrics, warnings = fetch_all_pair_metrics(pairs)

    for col in ["vol_30d_pct", "vol_1y_pct", "vol_5y_pct", "vol_g10_pct", "drift_pct"]:
        carry_df[col] = [
            metrics.get((long_ccy, short_ccy), {}).get(col)
            for long_ccy, short_ccy in zip(carry_df["long_ccy"], carry_df["short_ccy"])
        ]
    carry_df = carry_df.dropna(subset=["vol_g10_pct", "vol_5y_pct"], how="all").copy()

    depreciation_pct = (-carry_df["drift_pct"]).clip(lower=0)
    carry_df["adj_carry_pct"] = carry_df["carry_pct"] - depreciation_pct

    carry_df["carry_to_vol_g10"] = carry_df["carry_pct"] / carry_df["vol_g10_pct"]
    carry_df["adj_carry_to_vol_em"] = carry_df["adj_carry_pct"] / carry_df["vol_5y_pct"]

    carry_df["signal_g10"] = carry_df["carry_to_vol_g10"].apply(
        lambda v: _signal_for(v) if pd.notna(v) else None
    )
    carry_df["signal_em"] = carry_df["adj_carry_to_vol_em"].apply(
        lambda v: _signal_for(v) if pd.notna(v) else None
    )

    carry_df = carry_df.reset_index(drop=True)
    return carry_df, warnings
