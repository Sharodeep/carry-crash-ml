"""Walk-forward training, prediction, and SHAP interpretability for the
carry-crash early-warning models.

Expanding window, retrained at the start of each calendar year from 2008
onward, with a 4-week embargo dropped from the end of each training
window. The crash-label percentile threshold is recomputed per fold, using
only that fold's (post-embargo) training window.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CRASH_LABEL_PERCENTILE,
    GBM_PARAMS,
    LOGIT_PARAMS,
    WALK_FORWARD_EMBARGO_WEEKS,
    WALK_FORWARD_FIRST_TEST_YEAR,
)
from research.labels import label_crash, percentile_threshold


def make_logit() -> "sklearn.pipeline.Pipeline":
    return make_pipeline(StandardScaler(), LogisticRegression(**LOGIT_PARAMS))


def make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**GBM_PARAMS)


def walk_forward_folds(index: pd.DatetimeIndex, first_test_year: int = WALK_FORWARD_FIRST_TEST_YEAR):
    """Yield (train_index, test_index, test_year) for each expanding-window fold."""
    last_year = index.year.max()
    for test_year in range(first_test_year, last_year + 1):
        train_idx = index[index.year < test_year]
        test_idx = index[index.year == test_year]
        if len(train_idx) <= WALK_FORWARD_EMBARGO_WEEKS or test_idx.empty:
            continue
        yield train_idx, test_idx, test_year


def run_walk_forward(features: pd.DataFrame, F: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Returns (oos_predictions, fold_models).

    oos_predictions: DataFrame indexed by date, columns [logit_prob, gbm_prob,
    label, test_year, threshold], pooled across folds, restricted to weeks
    with a valid forward-return label.

    fold_models: dict[test_year] -> {"logit", "gbm", "X_train", "y_train",
    "threshold"}, so later steps (e.g. SHAP on the final fold) can reuse a
    fold's fitted models without refitting.
    """
    rows = []
    fold_models = {}

    for train_idx, test_idx, test_year in walk_forward_folds(features.index):
        effective_train_idx = train_idx[:-WALK_FORWARD_EMBARGO_WEEKS]

        X_train = features.loc[effective_train_idx]
        F_train = F.loc[effective_train_idx]
        valid_train = X_train.notna().all(axis=1) & F_train.notna()
        X_train, F_train = X_train[valid_train], F_train[valid_train]
        if X_train.empty:
            continue

        threshold = percentile_threshold(F_train, CRASH_LABEL_PERCENTILE)
        y_train = label_crash(F_train, threshold)
        if y_train.nunique() < 2:
            continue

        X_test = features.loc[test_idx]
        F_test = F.loc[test_idx]
        valid_test = X_test.notna().all(axis=1) & F_test.notna()
        X_test, F_test = X_test[valid_test], F_test[valid_test]
        if X_test.empty:
            continue
        y_test = label_crash(F_test, threshold)

        logit = make_logit()
        logit.fit(X_train, y_train)
        gbm = make_gbm()
        gbm.fit(X_train, y_train)

        logit_prob = logit.predict_proba(X_test)[:, 1]
        gbm_prob = gbm.predict_proba(X_test)[:, 1]

        fold_df = pd.DataFrame(
            {
                "logit_prob": logit_prob,
                "gbm_prob": gbm_prob,
                "label": y_test.values,
                "test_year": test_year,
                "threshold": threshold,
            },
            index=X_test.index,
        )
        rows.append(fold_df)

        fold_models[test_year] = {
            "logit": logit,
            "gbm": gbm,
            "X_train": X_train,
            "y_train": y_train,
            "threshold": threshold,
        }

    oos = pd.concat(rows).sort_index() if rows else pd.DataFrame()
    return oos, fold_models


def evaluate_pooled(oos: pd.DataFrame) -> pd.DataFrame:
    base_rate = oos["label"].mean()
    rows = []
    for model_name, prob_col in [("logit", "logit_prob"), ("gbm", "gbm_prob")]:
        rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(oos["label"], oos[prob_col]),
                "pr_auc": average_precision_score(oos["label"], oos[prob_col]),
                "base_rate": base_rate,
                "brier_score": brier_score_loss(oos["label"], oos[prob_col]),
            }
        )
    return pd.DataFrame(rows)


def final_fold(fold_models: dict) -> dict:
    """The most recent walk-forward fold's fitted models and training data."""
    return fold_models[max(fold_models)]


def compute_shap_global(gbm: HistGradientBoostingClassifier, X_train: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer(X_train)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return pd.Series(mean_abs, index=X_train.columns).sort_values(ascending=False)


def compute_shap_local(
    gbm: HistGradientBoostingClassifier, X_train: pd.DataFrame, row: pd.DataFrame
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer(row)
    contributions = pd.DataFrame(
        {
            "feature": row.columns,
            "value": row.iloc[0].values,
            "shap": shap_values.values[0],
        }
    )
    contributions["direction"] = np.where(contributions["shap"] > 0, "raises", "lowers")
    return contributions.reindex(contributions["shap"].abs().sort_values(ascending=False).index)


def compute_logit_local_drivers(logit: "sklearn.pipeline.Pipeline", row: pd.DataFrame) -> pd.DataFrame:
    """Linear analogue of compute_shap_local: contribution = coef * scaled_value.

    Logit is primary for the live regime panel (see outputs/results_summary.md
    for why), so this is what current_reading.py uses for top_drivers instead
    of SHAP, which only applies to the tree model.
    """
    scaler = logit.named_steps["standardscaler"]
    clf = logit.named_steps["logisticregression"]
    scaled = scaler.transform(row)[0]
    contributions = pd.DataFrame(
        {
            "feature": row.columns,
            "value": row.iloc[0].values,
            "shap": scaled * clf.coef_[0],
        }
    )
    contributions["direction"] = np.where(contributions["shap"] > 0, "raises", "lowers")
    return contributions.reindex(contributions["shap"].abs().sort_values(ascending=False).index)


def main():
    import os

    from research.build import build_carry_basket, build_weekly_panel, compute_excess_returns
    from research.features import build_features
    from research.labels import forward_return

    panel = build_weekly_panel()
    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])
    features = build_features(panel["fx_weekly"], panel["rate_weekly"], panel["risk_weekly"], basket)
    F = forward_return(basket["b_t"])

    oos, fold_models = run_walk_forward(features, F)

    os.makedirs("outputs/tables", exist_ok=True)
    oos.to_csv("outputs/tables/oos_predictions.csv")

    metrics = evaluate_pooled(oos)
    metrics.to_csv("outputs/tables/model_metrics.csv", index=False)

    print(f"Walk-forward folds: {sorted(fold_models.keys())}")
    print(f"Pooled OOS weeks: {len(oos)} ({oos.index.min().date()} -> {oos.index.max().date()})")
    print(f"\nModel metrics:\n{metrics.to_string(index=False)}")

    final = final_fold(fold_models)
    print(f"\nFinal fold (test_year={max(fold_models)}): training rows={len(final['X_train'])}")

    global_shap = compute_shap_global(final["gbm"], final["X_train"])
    print("\nGlobal mean |SHAP| (final fold GBM):")
    print(global_shap.to_string())

    week = pd.Timestamp("2024-08-02")
    if week in features.index:
        row = features.loc[[week]]
        if row.notna().all(axis=1).iloc[0]:
            local = compute_shap_local(final["gbm"], final["X_train"], row)
            print(f"\nLocal SHAP for week {week.date()} (final fold GBM):")
            print(local.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
