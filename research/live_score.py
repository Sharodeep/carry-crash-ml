"""Phase 2 bridge between the offline research pipeline and the live app.

Builds the latest week's crash features live (cheap: a few FRED pulls plus
pure pandas/numpy) and scores them with the models trained offline by
research/run_all.py. Does not retrain anything -- retraining is manual, via
run_all.py, with new model files committed afterward.

Logit is the primary model for the regime panel (see outputs/results_summary.md
for why): GBM's pooled OOS ROC AUC is below 0.5 and its probability line goes
flat for multi-year stretches, so regime/percentile/top_drivers are computed
from logit_prob. gbm_prob is still returned for the app to show as a secondary
figure if it wants to.
"""

import json
import os

import pandas as pd
from joblib import load

from config import MODELS_DIR, REGIME_ELEVATED_PERCENTILE, REGIME_HIGH_PERCENTILE, TABLES_DIR
from research.build import build_carry_basket, build_weekly_panel, compute_excess_returns
from research.features import build_features
from research.model import compute_logit_local_drivers


class ModelsNotFoundError(Exception):
    """Raised when models/ hasn't been populated by research/run_all.py yet."""


def _regime_from_percentile(percentile: float) -> str:
    if percentile >= REGIME_HIGH_PERCENTILE:
        return "High"
    if percentile >= REGIME_ELEVATED_PERCENTILE:
        return "Elevated"
    return "Calm"


def models_available() -> bool:
    return all(
        os.path.exists(os.path.join(MODELS_DIR, name))
        for name in ("crash_logit.joblib", "crash_gbm.joblib", "metadata.json")
    )


def load_models():
    if not models_available():
        raise ModelsNotFoundError(
            f"No trained models in {MODELS_DIR}/ -- run `python -m research.run_all` first."
        )
    logit = load(os.path.join(MODELS_DIR, "crash_logit.joblib"))
    gbm = load(os.path.join(MODELS_DIR, "crash_gbm.joblib"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return logit, gbm, metadata


def score_latest(refresh: bool = False) -> dict:
    """Builds the latest complete feature row and scores it live.

    Raises on any failure (missing models, FRED/network issues, an
    incomplete latest row) -- callers should catch and fall back to the
    committed outputs/current_reading.json.
    """
    logit, gbm, metadata = load_models()

    panel = build_weekly_panel(refresh=refresh)
    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])
    features = build_features(panel["fx_weekly"], panel["rate_weekly"], panel["risk_weekly"], basket)
    features = features[metadata["feature_list"]]

    complete_mask = features.notna().all(axis=1)
    if not complete_mask.any():
        raise ValueError("No complete feature row available to score.")
    latest_week = features[complete_mask].index.max()
    latest_row = features.loc[[latest_week]]

    logit_prob = float(logit.predict_proba(latest_row)[0, 1])
    gbm_prob = float(gbm.predict_proba(latest_row)[0, 1])

    oos_path = os.path.join(TABLES_DIR, "oos_predictions.csv")
    oos = pd.read_csv(oos_path, index_col=0, parse_dates=True)
    percentile_vs_history = float((oos["logit_prob"] < logit_prob).mean() * 100)
    regime = _regime_from_percentile(percentile_vs_history)

    local_drivers = compute_logit_local_drivers(logit, latest_row)
    top_drivers = [
        {"feature": row.feature, "value": float(row.value), "direction": row.direction}
        for row in local_drivers.head(3).itertuples()
    ]

    return {
        "as_of_week": latest_week.strftime("%Y-%m-%d"),
        "gbm_prob": round(gbm_prob, 4),
        "logit_prob": round(logit_prob, 4),
        "percentile_vs_history": round(percentile_vs_history, 1),
        "regime": regime,
        "top_drivers": top_drivers,
        "training_end_date": metadata["training_end_date"],
        "live": True,
    }


def load_cached_reading() -> dict:
    """Falls back to the offline run_all.py snapshot when live scoring fails."""
    with open("outputs/current_reading.json") as f:
        reading = json.load(f)
    reading["live"] = False
    return reading


def load_regime_history() -> pd.DataFrame:
    path = os.path.join(TABLES_DIR, "regime_history.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)
