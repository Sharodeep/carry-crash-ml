"""Feature construction. Pure functions, reusable by the live app in Phase 2.

Every feature here uses only information available at the end of week t:
contemporaneous levels, or backward-looking changes/rolling windows. None
of them look at F_t (the forward label) or any future basket return.
"""

import numpy as np
import pandas as pd

from config import (
    FEATURE_BRENT_RET_WEEKS,
    FEATURE_MOM_LONG_WEEKS,
    FEATURE_SHORT_CHANGE_WEEKS,
    FEATURE_VOL_WINDOW_WEEKS,
)

FEATURE_COLUMNS = [
    "vix",
    "vix_chg4",
    "basket_vol12",
    "basket_vol_chg4",
    "basket_mom12",
    "basket_mom4",
    "carry_spread",
    "credit",
    "credit_chg4",
    "brent_ret12",
    "jpy_ret4",
]


def _carry_spread(rate_rank_basis: pd.DataFrame, basket: pd.DataFrame) -> pd.Series:
    spread = pd.Series(index=basket.index, dtype=float)
    for t in basket.index:
        longs, shorts = basket.at[t, "longs"], basket.at[t, "shorts"]
        if not isinstance(longs, list) or not isinstance(shorts, list):
            spread.at[t] = np.nan
            continue
        spread.at[t] = (
            rate_rank_basis.loc[t, longs].mean() - rate_rank_basis.loc[t, shorts].mean()
        )
    return spread


def build_features(
    fx_weekly: pd.DataFrame,
    rate_weekly: pd.DataFrame,
    risk_weekly: pd.DataFrame,
    basket: pd.DataFrame,
) -> pd.DataFrame:
    idx = basket.index
    b = basket["b_t"]

    vix = risk_weekly["vix"].reindex(idx)
    vix_chg4 = vix - vix.shift(FEATURE_SHORT_CHANGE_WEEKS)

    basket_vol12 = b.rolling(FEATURE_VOL_WINDOW_WEEKS).std() * np.sqrt(52)
    basket_vol_chg4 = basket_vol12 - basket_vol12.shift(FEATURE_SHORT_CHANGE_WEEKS)
    basket_mom12 = b.rolling(FEATURE_MOM_LONG_WEEKS).sum()
    basket_mom4 = b.rolling(FEATURE_SHORT_CHANGE_WEEKS).sum()

    rate_rank_basis = rate_weekly.shift(1)
    carry_spread = _carry_spread(rate_rank_basis, basket)

    credit = risk_weekly["baa10y"].reindex(idx)
    credit_chg4 = credit - credit.shift(FEATURE_SHORT_CHANGE_WEEKS)

    brent = risk_weekly["brent"].reindex(idx)
    brent_ret12 = np.log(brent / brent.shift(FEATURE_BRENT_RET_WEEKS))

    jpy = fx_weekly["JPY"].reindex(idx)
    jpy_ret4 = np.log(jpy / jpy.shift(FEATURE_SHORT_CHANGE_WEEKS))

    features = pd.DataFrame(
        {
            "vix": vix,
            "vix_chg4": vix_chg4,
            "basket_vol12": basket_vol12,
            "basket_vol_chg4": basket_vol_chg4,
            "basket_mom12": basket_mom12,
            "basket_mom4": basket_mom4,
            "carry_spread": carry_spread,
            "credit": credit,
            "credit_chg4": credit_chg4,
            "brent_ret12": brent_ret12,
            "jpy_ret4": jpy_ret4,
        },
        index=idx,
    )
    return features[FEATURE_COLUMNS]


def main():
    from research.build import build_carry_basket, build_weekly_panel, compute_excess_returns

    panel = build_weekly_panel()
    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])

    features = build_features(panel["fx_weekly"], panel["rate_weekly"], panel["risk_weekly"], basket)

    print(f"Feature frame: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Usable rows (no NaN in any feature): {features.dropna().shape[0]}")
    print("\nPer-column NaN counts (expected at the start, from rolling windows):")
    print(features.isna().sum().to_string())

    print("\nDescribe:")
    print(features.describe().T[["mean", "std", "min", "max"]].to_string())

    print("\nWeek of 2024-08-02 (before the yen carry unwind):")
    nearest = features.index[features.index.get_indexer(["2024-08-02"], method="nearest")[0]]
    print(f"  (nearest available week: {nearest.date()})")
    print(features.loc[[nearest]].T.to_string())


if __name__ == "__main__":
    main()
