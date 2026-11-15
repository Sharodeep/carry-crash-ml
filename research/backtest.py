"""Strategy comparison: unconditional carry, vol-targeted carry (the naive
benchmark the ML must beat), and logit/GBM crash-signal strategies.

Weekly, out-of-sample period only (2008 onward, same pooled set as
model.py's walk-forward). 5bp transaction cost per unit of weight change,
applied uniformly across all strategies.
"""

import numpy as np
import pandas as pd

from config import (
    BACKTEST_NAMED_EPISODES,
    BACKTEST_SIGNAL_THRESHOLD_TOP_PCT,
    BACKTEST_TRANSACTION_COST_BP,
    BACKTEST_VOL_TARGET_WEIGHT_CAP,
)


def compute_weights(oos: pd.DataFrame, basket_b: pd.Series, basket_vol12: pd.Series, fold_models: dict) -> pd.DataFrame:
    idx = oos.index
    vol12_lag = basket_vol12.shift(1).reindex(idx)

    weights = pd.DataFrame(index=idx)
    weights["unconditional"] = 1.0

    target_vol_per_week = pd.Series(index=idx, dtype=float)
    for year, fm in fold_models.items():
        mask = oos["test_year"] == year
        train_b = basket_b.loc[fm["X_train"].index]
        target_vol = train_b.std() * np.sqrt(52)
        target_vol_per_week.loc[idx[mask]] = target_vol
    weights["vol_targeted"] = (target_vol_per_week / vol12_lag).clip(lower=0, upper=BACKTEST_VOL_TARGET_WEIGHT_CAP)

    for model_name, prob_col in [("logit", "logit_prob"), ("gbm", "gbm_prob")]:
        binary_w = pd.Series(index=idx, dtype=float)
        continuous_w = pd.Series(index=idx, dtype=float)
        for year, fm in fold_models.items():
            mask = oos["test_year"] == year
            test_idx_year = idx[mask]
            train_probs = fm[model_name].predict_proba(fm["X_train"])[:, 1]
            sig_threshold = np.quantile(train_probs, 1 - BACKTEST_SIGNAL_THRESHOLD_TOP_PCT)
            test_probs = oos.loc[test_idx_year, prob_col]
            binary_w.loc[test_idx_year] = (test_probs < sig_threshold).astype(float)
            continuous_w.loc[test_idx_year] = (1 - test_probs).clip(0, 1)
        weights[f"{model_name}_signal"] = binary_w
        weights[f"{model_name}_continuous"] = continuous_w

    return weights.fillna(0.0)


def compute_net_returns(weights: pd.DataFrame, basket_b: pd.Series) -> pd.DataFrame:
    idx = weights.index
    b = basket_b.reindex(idx)
    cost_rate = BACKTEST_TRANSACTION_COST_BP / 10000.0

    returns = {}
    for col in weights.columns:
        w = weights[col]
        turnover = w.diff().abs()
        turnover.iloc[0] = abs(w.iloc[0])
        cost = cost_rate * turnover
        returns[col] = w * b - cost

    return pd.DataFrame(returns, index=idx)


def compute_strategy_metrics(returns: pd.Series, weights: pd.Series) -> dict:
    ann_return = returns.mean() * 52
    ann_vol = returns.std() * np.sqrt(52)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cumulative = returns.cumsum()
    drawdown = cumulative - cumulative.cummax()

    metrics = {
        "annualised_return": ann_return,
        "annualised_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "worst_4wk_return": returns.rolling(4).sum().min(),
        "pct_weeks_invested": (weights.abs() > 1e-9).mean(),
    }

    for name, (start, end) in BACKTEST_NAMED_EPISODES.items():
        window = returns.loc[start:end]
        metrics[f"episode_{name}"] = window.sum() if not window.empty else np.nan

    return metrics


def run_backtest(oos: pd.DataFrame, basket_b: pd.Series, basket_vol12: pd.Series, fold_models: dict):
    weights = compute_weights(oos, basket_b, basket_vol12, fold_models)
    returns = compute_net_returns(weights, basket_b)

    metrics = pd.DataFrame(
        {col: compute_strategy_metrics(returns[col], weights[col]) for col in returns.columns}
    ).T
    metrics.index.name = "strategy"
    return weights, returns, metrics


def main():
    import os

    from research.build import build_carry_basket, build_weekly_panel, compute_excess_returns
    from research.features import build_features
    from research.labels import forward_return
    from research.model import run_walk_forward

    panel = build_weekly_panel()
    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])
    features = build_features(panel["fx_weekly"], panel["rate_weekly"], panel["risk_weekly"], basket)
    F = forward_return(basket["b_t"])

    oos, fold_models = run_walk_forward(features, F)
    basket_vol12 = features["basket_vol12"]

    weights, returns, metrics = run_backtest(oos, basket["b_t"], basket_vol12, fold_models)

    os.makedirs("outputs/tables", exist_ok=True)
    metrics.to_csv("outputs/tables/backtest_metrics.csv")
    returns.to_csv("outputs/tables/backtest_returns.csv")

    pd.set_option("display.width", 160)
    print(metrics.round(4).to_string())

    uncond_sharpe = metrics.loc["unconditional", "sharpe"]
    vol_sharpe = metrics.loc["vol_targeted", "sharpe"]
    best_signal = metrics.loc[["logit_signal", "gbm_signal", "logit_continuous", "gbm_continuous"], "sharpe"].idxmax()
    best_signal_sharpe = metrics.loc[best_signal, "sharpe"]

    print(f"\nUnconditional Sharpe: {uncond_sharpe:.3f}")
    print(f"Vol-targeted Sharpe (benchmark the ML must beat): {vol_sharpe:.3f}")
    print(f"Best ML-signal strategy: {best_signal} (Sharpe {best_signal_sharpe:.3f})")
    if best_signal_sharpe > vol_sharpe:
        print("-> ML-based signal beats the vol-targeting benchmark on this sample.")
    else:
        print("-> ML-based signal does NOT beat the vol-targeting benchmark on this sample.")


if __name__ == "__main__":
    main()
