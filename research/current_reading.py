"""Score the latest complete week: train final models on all labelled data,
save them for the live app, and write outputs/current_reading.json.

This is the bridge between the research pipeline and the Phase 2 app.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import shap

from config import (
    CRASH_LABEL_PERCENTILE,
    CURRENT_READING_STALE_INPUT_DAYS,
    MODELS_DIR,
    REGIME_ELEVATED_PERCENTILE,
    REGIME_HIGH_PERCENTILE,
    RESEARCH_FX_SERIES,
    RESEARCH_RATE_SERIES,
    RESEARCH_RISK_SERIES,
)
from research.labels import forward_return, label_crash, percentile_threshold
from research.model import compute_logit_local_drivers, make_gbm, make_logit


def _regime_from_percentile(percentile: float) -> str:
    if percentile >= REGIME_HIGH_PERCENTILE:
        return "High"
    if percentile >= REGIME_ELEVATED_PERCENTILE:
        return "Elevated"
    return "Calm"


def _input_dates_and_staleness(fred, refresh: bool = False):
    from research.fetch import fetch_raw_series

    dates = {}
    today = pd.Timestamp.today()

    for currency, info in RESEARCH_FX_SERIES.items():
        series = fetch_raw_series(fred, info["series_id"], refresh=refresh)
        dates[info["series_id"]] = series.index.max()

    for currency, info in RESEARCH_RATE_SERIES.items():
        primary = fetch_raw_series(fred, info["primary"], refresh=refresh)
        last = primary.index.max()
        if info["splice"] is not None:
            splice = fetch_raw_series(fred, info["splice"], refresh=refresh)
            last = max(last, splice.index.max())
        dates[info["primary"]] = last

    for name, series_id in RESEARCH_RISK_SERIES.items():
        series = fetch_raw_series(fred, series_id, refresh=refresh)
        dates[series_id] = series.index.max()

    stale = [
        series_id
        for series_id, last in dates.items()
        if (today - last).days > CURRENT_READING_STALE_INPUT_DAYS
    ]
    return dates, stale


def run_current_reading(refresh: bool = False) -> dict:
    from research.build import build_carry_basket, build_weekly_panel, compute_excess_returns
    from research.features import build_features
    from research.model import run_walk_forward

    panel = build_weekly_panel(refresh=refresh)
    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])
    features = build_features(panel["fx_weekly"], panel["rate_weekly"], panel["risk_weekly"], basket)
    F = forward_return(basket["b_t"])

    labelled_mask = F.notna() & features.notna().all(axis=1)
    X_train = features[labelled_mask]
    F_train = F[labelled_mask]
    threshold = percentile_threshold(F_train, CRASH_LABEL_PERCENTILE)
    y_train = label_crash(F_train, threshold)

    final_logit = make_logit()
    final_logit.fit(X_train, y_train)
    final_gbm = make_gbm()
    final_gbm.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_logit, os.path.join(MODELS_DIR, "crash_logit.joblib"))
    joblib.dump(final_gbm, os.path.join(MODELS_DIR, "crash_gbm.joblib"))

    # Historical OOS probability distribution (for regime percentiles) comes
    # from the walk-forward run, not from in-sample predictions. Logit is
    # primary (see outputs/results_summary.md: GBM's pooled OOS AUC is below
    # 0.5 and its probability line goes flat for multi-year stretches), so
    # regime/percentile/top_drivers are computed from logit_prob. GBM is kept
    # as a secondary/diagnostic field.
    oos, _ = run_walk_forward(features, F)
    oos_percentiles = {
        p: float(np.percentile(oos["logit_prob"], p)) for p in (10, 25, 50, 75, 90)
    }
    oos_percentiles_gbm = {
        p: float(np.percentile(oos["gbm_prob"], p)) for p in (10, 25, 50, 75, 90)
    }

    train_probs_gbm = final_gbm.predict_proba(X_train)[:, 1]
    train_probs_logit = final_logit.predict_proba(X_train)[:, 1]
    signal_threshold_gbm = float(np.quantile(train_probs_gbm, 0.80))
    signal_threshold_logit = float(np.quantile(train_probs_logit, 0.80))

    metadata = {
        "training_end_date": X_train.index.max().strftime("%Y-%m-%d"),
        "feature_list": list(features.columns),
        "primary_model": "logit",
        "crash_label_threshold": float(threshold),
        "signal_threshold_top20pct": {
            "gbm": signal_threshold_gbm,
            "logit": signal_threshold_logit,
        },
        "historical_oos_logit_prob_percentiles": oos_percentiles,
        "historical_oos_gbm_prob_percentiles": oos_percentiles_gbm,
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "shap": shap.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Latest complete week: most recent row with no NaN feature.
    complete_mask = features.notna().all(axis=1)
    latest_week = features[complete_mask].index.max()
    latest_row = features.loc[[latest_week]]

    gbm_prob = float(final_gbm.predict_proba(latest_row)[0, 1])
    logit_prob = float(final_logit.predict_proba(latest_row)[0, 1])

    percentile_vs_history = float(
        (oos["logit_prob"] < logit_prob).mean() * 100
    )
    regime = _regime_from_percentile(percentile_vs_history)

    local_drivers = compute_logit_local_drivers(final_logit, latest_row)
    top_drivers = [
        {"feature": row.feature, "value": float(row.value), "direction": row.direction}
        for row in local_drivers.head(3).itertuples()
    ]

    from research.fetch import get_fred_client

    fred = get_fred_client()
    input_dates, stale_inputs = _input_dates_and_staleness(fred, refresh=refresh)

    reading = {
        "as_of_week": latest_week.strftime("%Y-%m-%d"),
        "gbm_prob": round(gbm_prob, 4),
        "logit_prob": round(logit_prob, 4),
        "percentile_vs_history": round(percentile_vs_history, 1),
        "regime": regime,
        "top_drivers": top_drivers,
        "training_end_date": metadata["training_end_date"],
        "input_dates": {k: v.strftime("%Y-%m-%d") for k, v in input_dates.items()},
        "stale_inputs": stale_inputs,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/current_reading.json", "w") as f:
        json.dump(reading, f, indent=2)

    # Regime history for plots.py's figure 08 (last 3 years + latest week).
    # logit_prob is primary; gbm_prob kept alongside as a secondary/diagnostic column.
    regime_history = oos[["logit_prob", "gbm_prob"]].copy()
    regime_history["percentile_vs_history"] = regime_history["logit_prob"].rank(pct=True) * 100
    regime_history["regime"] = regime_history["percentile_vs_history"].apply(_regime_from_percentile)
    latest_row_hist = pd.DataFrame(
        {
            "logit_prob": [logit_prob],
            "gbm_prob": [gbm_prob],
            "percentile_vs_history": [percentile_vs_history],
            "regime": [regime],
        },
        index=[latest_week],
    )
    regime_history = pd.concat([regime_history, latest_row_hist]).sort_index()
    os.makedirs("outputs/tables", exist_ok=True)
    regime_history.to_csv("outputs/tables/regime_history.csv")

    return reading


def main():
    reading = run_current_reading()
    print(json.dumps(reading, indent=2))

    if reading["stale_inputs"]:
        print(f"\nWARNING: {len(reading['stale_inputs'])} input series are stale (> "
              f"{CURRENT_READING_STALE_INPUT_DAYS} days): {reading['stale_inputs']}")


if __name__ == "__main__":
    main()
