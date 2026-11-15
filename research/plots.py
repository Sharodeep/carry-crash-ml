"""Slide-ready figures for the course presentation. White background, no
chart junk, font 12+, figsize (10, 5.6), 200 dpi, saved to outputs/figures/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from config import BACKTEST_NAMED_EPISODES, FIGURES_DIR
from research.model import compute_shap_global, compute_shap_local, final_fold

plt.rcParams.update({"font.size": 12, "figure.facecolor": "white", "axes.facecolor": "white"})

FIGSIZE = (10, 5.6)
DPI = 200


def _new_fig():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=DPI)
    plt.close(fig)


def plot_basket_cumulative(basket: pd.DataFrame):
    cumulative = basket["b_t"].cumsum()
    fig, ax = _new_fig()
    ax.plot(cumulative.index, cumulative.values, color="#1f4e8c", linewidth=1.4)
    ymin, ymax = cumulative.min(), cumulative.max()
    headroom = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - headroom * 0.3, ymax + headroom)
    for name, (start, end) in BACKTEST_NAMED_EPISODES.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#d62728", alpha=0.15)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.annotate(name, xy=(mid, ymax + headroom * 0.15), xytext=(mid, ymax + headroom * 0.15),
                    rotation=90, fontsize=9, ha="center", va="top", color="#8b0000")
    ax.set_title("G10 carry basket: cumulative log excess return")
    ax.set_ylabel("Cumulative log excess return")
    _save(fig, "01_basket_cumulative.png")


def plot_crash_probability(oos: pd.DataFrame):
    fig, ax = _new_fig()
    ax.plot(oos.index, oos["gbm_prob"], color="#1f4e8c", linewidth=1.3, label="GBM probability")
    ax.plot(oos.index, oos["logit_prob"], color="#999999", linewidth=0.8, label="Logit probability")
    crash_weeks = oos[oos["label"] == 1]
    ax.scatter(crash_weeks.index, crash_weeks["gbm_prob"], color="#d62728", s=14, zorder=5, label="Actual crash week")
    ax.set_title("Out-of-sample 4-week crash probability")
    ax.set_ylabel("Predicted probability")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "02_crash_probability.png")


def plot_roc_pr(oos: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    base_rate = oos["label"].mean()

    for model_name, prob_col, color in [("Logit", "logit_prob", "#1f4e8c"), ("GBM", "gbm_prob", "#d62728")]:
        fpr, tpr, _ = roc_curve(oos["label"], oos[prob_col])
        ax1.plot(fpr, tpr, label=model_name, color=color, linewidth=1.4)
        precision, recall, _ = precision_recall_curve(oos["label"], oos[prob_col])
        ax2.plot(recall, precision, label=model_name, color=color, linewidth=1.4)

    ax1.plot([0, 1], [0, 1], linestyle="--", color="#bbbbbb", linewidth=1)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.axhline(base_rate, linestyle="--", color="#bbbbbb", linewidth=1, label="Base rate")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    _save(fig, "03_roc_pr.png")


def plot_backtest_cumulative(returns: pd.DataFrame):
    fig, ax = _new_fig()
    strategies = {
        "unconditional": ("Unconditional carry", "#999999"),
        "vol_targeted": ("Vol-targeted carry", "#f5a623"),
        "logit_signal": ("Logit signal", "#1f4e8c"),
        "gbm_signal": ("GBM signal", "#d62728"),
    }
    for col, (label, color) in strategies.items():
        cumulative = returns[col].cumsum()
        ax.plot(cumulative.index, cumulative.values, label=label, color=color, linewidth=1.4)
    ax.set_title("Backtest: cumulative return by strategy (2008 onward)")
    ax.set_ylabel("Cumulative log return (net of transaction costs)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "04_backtest_cumulative.png")


def plot_drawdowns(returns: pd.DataFrame, metrics: pd.DataFrame):
    signal_cols = ["logit_signal", "gbm_signal", "logit_continuous", "gbm_continuous"]
    best_signal = metrics.loc[signal_cols, "sharpe"].idxmax()

    fig, ax = _new_fig()
    for col, label, color in [
        ("unconditional", "Unconditional carry", "#999999"),
        (best_signal, f"Best signal ({best_signal})", "#1f4e8c"),
    ]:
        cumulative = returns[col].cumsum()
        drawdown = cumulative - cumulative.cummax()
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color=color)
        ax.plot(drawdown.index, drawdown.values, color=color, linewidth=1.2, label=label)
    ax.set_title("Drawdowns: unconditional vs. best signal strategy")
    ax.set_ylabel("Drawdown (cumulative log return)")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "05_drawdowns.png")


def plot_shap_global(gbm, X_train: pd.DataFrame):
    global_shap = compute_shap_global(gbm, X_train)
    fig, ax = _new_fig()
    ax.barh(global_shap.index[::-1], global_shap.values[::-1], color="#1f4e8c")
    ax.set_title("Global feature importance (mean |SHAP|, final-fold GBM)")
    ax.set_xlabel("Mean |SHAP value|")
    _save(fig, "06_shap_global.png")


def plot_shap_local(gbm, X_train: pd.DataFrame, features: pd.DataFrame):
    week = pd.Timestamp("2024-08-02")
    row = features.loc[[week]]
    local = compute_shap_local(gbm, X_train, row)

    fig, ax = _new_fig()
    colors = ["#d62728" if v > 0 else "#1f4e8c" for v in local["shap"]]
    ax.barh(local["feature"][::-1], local["shap"][::-1], color=colors[::-1])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Local SHAP explanation: week of {week.date()} (before the yen carry unwind)")
    ax.set_xlabel("SHAP value (red = raises crash probability)")
    _save(fig, "07_shap_aug2024.png")


def plot_current_reading(regime_history: pd.DataFrame):
    recent = regime_history.loc[regime_history.index >= regime_history.index.max() - pd.Timedelta(weeks=156)]

    regime_colors = {"Calm": "#00c896", "Elevated": "#f5a623", "High": "#e05c5c"}
    fig, ax = _new_fig()

    if "gbm_prob" in recent.columns:
        ax.plot(recent.index, recent["gbm_prob"], color="#bbbbbb", linewidth=0.8, alpha=0.7,
                 zorder=1, label="GBM (secondary)")

    for regime, color in regime_colors.items():
        mask = recent["regime"] == regime
        ax.scatter(recent.index[mask], recent.loc[mask, "logit_prob"], color=color, s=10, label=regime)

    ax.plot(recent.index, recent["logit_prob"], color="#333333", linewidth=0.6, alpha=0.5, zorder=1)

    latest = recent.iloc[-1]
    ax.scatter([recent.index[-1]], [latest["logit_prob"]], color="black", s=60, zorder=6, marker="D")
    ax.annotate(
        f"{latest['logit_prob']:.2f}",
        xy=(recent.index[-1], latest["logit_prob"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title("Current carry-crash reading (logit primary, last 3 years)")
    ax.set_ylabel("Logit probability")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "08_current_reading.png")


def main():
    from research.backtest import run_backtest
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
    weights, returns, metrics = run_backtest(oos, basket["b_t"], features["basket_vol12"], fold_models)

    final = final_fold(fold_models)

    plot_basket_cumulative(basket)
    plot_crash_probability(oos)
    plot_roc_pr(oos)
    plot_backtest_cumulative(returns)
    plot_drawdowns(returns, metrics)
    plot_shap_global(final["gbm"], final["X_train"])
    plot_shap_local(final["gbm"], final["X_train"], features)

    regime_history_path = "outputs/tables/regime_history.csv"
    if os.path.exists(regime_history_path):
        regime_history = pd.read_csv(regime_history_path, index_col=0, parse_dates=True)
        plot_current_reading(regime_history)
    else:
        print("Skipping 08_current_reading.png -- run research/current_reading.py first.")

    print(f"Figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
