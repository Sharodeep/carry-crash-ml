"""Crash label: forward 4-week basket return below a percentile threshold.

Pure functions only -- the percentile threshold must be recomputed per
walk-forward fold (training window only) by model.py. This module does not
decide what counts as "the" threshold; it just computes forward returns and
applies whatever threshold it's given.
"""

import pandas as pd

from config import CRASH_FORWARD_WEEKS, CRASH_LABEL_PERCENTILE, CRASH_ROBUSTNESS_THRESHOLD


def forward_return(b: pd.Series, weeks: int = CRASH_FORWARD_WEEKS) -> pd.Series:
    """F_t = sum(b_t+1 ... b_t+weeks). NaN for the last `weeks` rows (no label yet)."""
    cumsum = b.cumsum()
    return cumsum.shift(-weeks) - cumsum


def percentile_threshold(F_train: pd.Series, percentile: int = CRASH_LABEL_PERCENTILE) -> float:
    """Threshold computed on a training window only -- never the full sample."""
    return F_train.dropna().quantile(percentile / 100.0)


def label_crash(F: pd.Series, threshold: float) -> pd.Series:
    return (F < threshold).astype(float).where(F.notna())


def label_crash_robust(F: pd.Series, threshold: float = CRASH_ROBUSTNESS_THRESHOLD) -> pd.Series:
    return (F < threshold).astype(float).where(F.notna())


def crash_frequency_by_year(labels: pd.Series) -> pd.Series:
    valid = labels.dropna()
    return valid.groupby(valid.index.year).mean()


def main():
    basket = pd.read_csv("outputs/tables/basket_weekly.csv", index_col=0, parse_dates=True)
    F = forward_return(basket["b_t"])

    full_sample_threshold = percentile_threshold(F)
    primary_labels = label_crash(F, full_sample_threshold)
    robust_labels = label_crash_robust(F)

    print(
        "NOTE: this is a full-sample diagnostic only. The labels actually used "
        "for training recompute the percentile threshold per walk-forward fold "
        "(training window only), per config.CRASH_LABEL_PERCENTILE."
    )
    print(f"\nFull-sample F_t 10th percentile threshold: {full_sample_threshold:+.4f}")
    print(f"Primary label positive rate (full sample): {primary_labels.mean():.1%}")
    print(f"Robustness label positive rate (F_t < {CRASH_ROBUSTNESS_THRESHOLD:.0%}): {robust_labels.mean():.1%}")

    print("\nCrash frequency by year (primary label, full-sample threshold):")
    print(crash_frequency_by_year(primary_labels).to_string())

    print("\nCrash frequency by year (robustness label, F_t < -3%):")
    print(crash_frequency_by_year(robust_labels).to_string())

    known_crash_years = [2008, 2011, 2015, 2020, 2024]
    freq = crash_frequency_by_year(primary_labels)
    print("\nCheck: known crash years should show elevated frequency vs. the ~10% base rate:")
    for year in known_crash_years:
        if year in freq.index:
            print(f"  {year}: {freq[year]:.1%}")


if __name__ == "__main__":
    main()
