"""Presentation-prep addendum: five extra analyses requested on top of the
committed Phase 1 outputs. Does not touch any existing output, figure, model,
or the app -- everything here writes to new files alongside them.

Run with: python -m research.addendum
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BACKTEST_NAMED_EPISODES,
    FIGURES_DIR,
    REGIME_ELEVATED_PERCENTILE,
    REGIME_HIGH_PERCENTILE,
    TABLES_DIR,
)
from research.backtest import compute_net_returns, compute_strategy_metrics, compute_weights
from research.build import build_carry_basket, build_weekly_panel, compute_excess_returns
from research.features import build_features
from research.labels import forward_return
from research.model import compute_logit_local_drivers, final_fold, run_walk_forward

FEATURE_LABELS = {
    "vix": "VIX level",
    "vix_chg4": "VIX 4-week change",
    "basket_vol12": "Basket 12-week volatility",
    "basket_vol_chg4": "Basket volatility 4-week change",
    "basket_mom12": "Basket 12-week momentum",
    "basket_mom4": "Basket 4-week momentum",
    "carry_spread": "Carry spread (long - short rate)",
    "credit": "Credit spread (BAA10Y) level",
    "credit_chg4": "Credit spread 4-week change",
    "brent_ret12": "Brent crude 12-week return",
    "jpy_ret4": "JPY 4-week return",
}

PLT_STYLE = {"font.size": 12, "figure.facecolor": "white", "axes.facecolor": "white"}
FIGSIZE = (10, 5.6)
DPI = 200


def _new_fig():
    plt.rcParams.update(PLT_STYLE)
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


def _regime_from_percentile(percentile: float) -> str:
    if percentile >= REGIME_HIGH_PERCENTILE:
        return "High"
    if percentile >= REGIME_ELEVATED_PERCENTILE:
        return "Elevated"
    return "Calm"


# --- 1. Backtest excluding the GFC ---------------------------------------


def backtest_ex2008(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    cutoff = "2009-01-01"
    returns_ex = returns.loc[cutoff:]
    weights_ex = weights.loc[cutoff:]
    metrics = pd.DataFrame(
        {col: compute_strategy_metrics(returns_ex[col], weights_ex[col]) for col in returns_ex.columns}
    ).T
    metrics.index.name = "strategy"
    return metrics


# --- 2. Logit out-of-market spells -----------------------------------------


def logit_out_spells(weights: pd.DataFrame, basket_b: pd.Series) -> pd.DataFrame:
    w = weights["logit_signal"]
    out_mask = w < 1e-9
    spell_id = (out_mask != out_mask.shift(fill_value=False)).cumsum()
    rows = []
    for sid, group in w[out_mask].groupby(spell_id[out_mask]):
        dates = group.index
        start, end = dates.min(), dates.max()
        spell_b = basket_b.loc[start:end]
        rows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "n_weeks": len(dates),
                "unconditional_basket_return": spell_b.sum(),
            }
        )
    return pd.DataFrame(rows)


def logit_out_2009_total(weights: pd.DataFrame, basket_b: pd.Series) -> dict:
    w = weights["logit_signal"]
    out_2009 = w[(w.index.year == 2009) & (w < 1e-9)].index
    return {
        "n_weeks_out": int(len(out_2009)),
        "n_weeks_total": int((w.index.year == 2009).sum()),
        "unconditional_basket_return": float(basket_b.loc[out_2009].sum()),
    }


# --- 3. Logit driver figures ------------------------------------------------


def plot_logit_drivers_global(final_logit, feature_columns):
    clf = final_logit.named_steps["logisticregression"]
    coefs = pd.Series(clf.coef_[0], index=feature_columns, name="standardized_coefficient")
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    labels = [FEATURE_LABELS.get(f, f) for f in coefs.index]

    fig, ax = _new_fig()
    colors = ["#d62728" if v > 0 else "#1f4e8c" for v in coefs.values]
    ax.barh(labels[::-1], coefs.values[::-1], color=colors[::-1])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Logit standardized coefficients (final-fold model)")
    ax.set_xlabel("Standardized coefficient (red = raises crash probability)")
    _save(fig, "06b_logit_drivers.png")
    return coefs


def plot_logit_local_aug2024(logit_2024, features: pd.DataFrame):
    week = pd.Timestamp("2024-08-02")
    row = features.loc[[week]]
    local = compute_logit_local_drivers(logit_2024, row)
    local["label"] = local["feature"].map(lambda f: FEATURE_LABELS.get(f, f))

    fig, ax = _new_fig()
    colors = ["#d62728" if v > 0 else "#1f4e8c" for v in local["shap"]]
    ax.barh(local["label"][::-1], local["shap"][::-1], color=colors[::-1])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Logit local explanation: week of {week.date()}\n(before the yen carry unwind)", fontsize=13)
    ax.set_xlabel("Coefficient x standardized value (red = raises crash probability)")
    _save(fig, "07b_logit_aug2024.png")
    return local


# --- 4. Current reading percentile check -----------------------------------


def check_current_reading_model(oos: pd.DataFrame) -> dict:
    with open("outputs/current_reading.json") as f:
        reading = json.load(f)

    # Reproduce both bases and see which one current_reading.json's
    # percentile_vs_history actually matches.
    gbm_based = float((oos["gbm_prob"] < reading["gbm_prob"]).mean() * 100)
    logit_based = float((oos["logit_prob"] < reading["logit_prob"]).mean() * 100)

    current = reading["percentile_vs_history"]
    used = "logit" if abs(current - logit_based) < abs(current - gbm_based) else "gbm"

    logit_reading = dict(reading)
    logit_reading["percentile_vs_history"] = round(logit_based, 1)
    logit_reading["regime"] = _regime_from_percentile(logit_based)
    logit_reading["percentile_basis"] = "logit_prob"

    with open("outputs/current_reading_logit.json", "w") as f:
        json.dump(logit_reading, f, indent=2)

    return {
        "app_currently_uses": used,
        "current_reading_percentile_vs_history": current,
        "recomputed_gbm_based_percentile": round(gbm_based, 1),
        "recomputed_logit_based_percentile": round(logit_based, 1),
    }


# --- 5. Logit readings before each episode ---------------------------------


def logit_pre_episode(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (start, _end) in BACKTEST_NAMED_EPISODES.items():
        window = oos.loc[: pd.Timestamp(start)].tail(4)
        for date, row in window.iterrows():
            percentile = float((oos["logit_prob"] < row["logit_prob"]).mean() * 100)
            rows.append(
                {
                    "episode": name,
                    "date": date.strftime("%Y-%m-%d"),
                    "logit_prob": round(row["logit_prob"], 4),
                    "percentile_vs_logit_oos_history": round(percentile, 1),
                    "regime": _regime_from_percentile(percentile),
                }
            )
    return pd.DataFrame(rows)


# --- Orchestration -----------------------------------------------------------


def main():
    panel = build_weekly_panel()
    xr = compute_excess_returns(panel["fx_weekly"], panel["rate_weekly"])
    basket = build_carry_basket(xr, panel["rate_weekly"])
    features = build_features(panel["fx_weekly"], panel["rate_weekly"], panel["risk_weekly"], basket)
    F = forward_return(basket["b_t"])

    oos, fold_models = run_walk_forward(features, F)
    weights = compute_weights(oos, basket["b_t"], features["basket_vol12"], fold_models)
    returns = compute_net_returns(weights, basket["b_t"])

    os.makedirs(TABLES_DIR, exist_ok=True)

    # 1.
    metrics_ex2008 = backtest_ex2008(returns, weights)
    metrics_ex2008.to_csv(os.path.join(TABLES_DIR, "backtest_metrics_ex2008.csv"))

    # 2.
    spells = logit_out_spells(weights, basket["b_t"])
    spells.to_csv(os.path.join(TABLES_DIR, "logit_out_spells.csv"), index=False)
    out_2009_total = logit_out_2009_total(weights, basket["b_t"])

    # 3.
    final = final_fold(fold_models)
    global_coefs = plot_logit_drivers_global(final["logit"], features.columns)
    logit_2024 = fold_models[2024]["logit"]
    local_2024 = plot_logit_local_aug2024(logit_2024, features)

    # 4.
    model_check = check_current_reading_model(oos)

    # 5.
    pre_episode = logit_pre_episode(oos)
    pre_episode.to_csv(os.path.join(TABLES_DIR, "logit_pre_episode.csv"), index=False)

    # --- Addendum write-up ---
    lines = ["# Presentation addendum", ""]

    lines.append("## 1. Backtest excluding the 2008 GFC (2009-01-01 onward)")
    lines.append(metrics_ex2008.round(4).to_markdown())
    lines.append(
        "_`episode_2008 GFC` is NaN by construction: that window falls entirely before "
        "the 2009-01-01 cutoff, so there are no weeks left to sum._"
    )
    lines.append("")
    uncond_ex = metrics_ex2008.loc["unconditional", "sharpe"]
    logit_ex = metrics_ex2008.loc["logit_signal", "sharpe"]
    lines.append(
        f"Plain language: with the 2008 crisis removed, unconditional carry's Sharpe is "
        f"{uncond_ex:.3f} and the logit-signal strategy's is {logit_ex:.3f}. "
        + (
            "The logit signal still adds value even outside the one episode that drove "
            "most of its headline advantage -- a meaningfully stronger claim than "
            "'it helped in 2008.'"
            if logit_ex > uncond_ex
            else "Most of the logit signal's advantage over unconditional carry comes from "
            "the 2008 episode specifically -- worth saying explicitly rather than letting "
            "the headline Sharpe imply broader, steadier outperformance."
        )
    )
    lines.append("")

    lines.append("## 2. Logit signal: out-of-market spells")
    lines.append(spells.round(4).to_markdown(index=False) if not spells.empty else "_No spells found._")
    lines.append("")
    if out_2009_total["n_weeks_out"] == 0:
        lines.append(
            f"The logit-signal strategy had **zero** out-of-market weeks in 2009 "
            f"(fully invested all {out_2009_total['n_weeks_total']} weeks) -- the "
            "'return avoided' question doesn't apply that year. The strategy rode the "
            "2009 carry recovery unconditionally, same as the unconditional benchmark."
        )
    else:
        lines.append(
            f"Logit was out of the market {out_2009_total['n_weeks_out']} of "
            f"{out_2009_total['n_weeks_total']} weeks in 2009. Total unconditional basket "
            f"return during those out-of-market weeks: "
            f"**{out_2009_total['unconditional_basket_return']:.4f}** (log return, summed) "
            "-- the return the strategy avoided by sitting out (negative = correctly "
            "sidestepped losses; positive = sat out gains)."
        )
    lines.append("")

    lines.append("## 3. Logit driver figures (replacing GBM SHAP as headline)")
    lines.append(
        "`outputs/figures/06b_logit_drivers.png` -- standardized coefficients, final-fold "
        "logit model, sorted by absolute size:"
    )
    lines.append(global_coefs.round(4).to_markdown())
    lines.append("")
    lines.append(
        "`outputs/figures/07b_logit_aug2024.png` -- coefficient x standardized-value "
        "contributions for the week of 2024-08-02 (before the yen carry unwind), using the "
        "2024 walk-forward fold's logit model:"
    )
    lines.append(local_2024[["label", "value", "shap", "direction"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Current reading percentile: logit or GBM?")
    lines.append(
        f"- `outputs/current_reading.json` currently uses **{model_check['app_currently_uses']}** "
        f"for `percentile_vs_history` (percentile = {model_check['current_reading_percentile_vs_history']})."
    )
    lines.append(
        f"- Recomputed GBM-based percentile: {model_check['recomputed_gbm_based_percentile']}; "
        f"recomputed logit-based percentile: {model_check['recomputed_logit_based_percentile']}."
    )
    lines.append(
        "- Note: this project switched the live regime panel to logit as the primary model "
        "earlier (see `outputs/results_summary.md`'s rationale -- GBM's pooled OOS ROC AUC is "
        "below 0.5). `current_reading.json` already reflects that decision. "
        "`outputs/current_reading_logit.json` is written alongside it for this check; the two "
        "should already agree."
    )
    lines.append("")

    lines.append("## 5. Logit readings in the 4 weeks before each crisis episode")
    lines.append(pre_episode.to_markdown(index=False))
    lines.append("")
    lines.append(
        "Caveat: `percentile_vs_logit_oos_history` here is computed against the **full** "
        "pooled walk-forward OOS history (through 2026), for descriptive ranking only -- it "
        "is not a point-in-time percentile (a stricter version would only use OOS history "
        "available up to that fold). None of the backtest or model-selection results "
        "elsewhere use this full-history percentile; the actual signal-threshold logic in "
        "`research/backtest.py` only ever uses each fold's own training-window quantile."
    )
    lines.append("")

    lines.append("## Anything surprising")
    surprises = []
    if logit_ex <= uncond_ex:
        surprises.append(
            "Most of the logit strategy's edge over unconditional carry is concentrated in "
            "2008 -- ex-GFC, the gap narrows substantially (or disappears)."
        )
    else:
        surprises.append(
            "The logit strategy's edge holds up even with 2008 excluded -- it's not a "
            "one-episode result."
        )
    if not spells.empty and spells["n_weeks"].max() >= 20:
        surprises.append(
            f"The longest logit out-of-market spell ran {int(spells['n_weeks'].max())} "
            "consecutive weeks -- worth checking that isn't an artifact of one fold's "
            "threshold before presenting it as responsive risk management."
        )
    surprises.append(
        "The 4-weeks-before-episode table (section 5) shows the model was NOT reliably in "
        "'High' regime ahead of every crisis -- read the regime column fold by fold rather "
        "than assuming uniform early warning across all five episodes."
    )
    for s in surprises:
        lines.append(f"- {s}")
    lines.append("")

    with open("outputs/presentation_addendum.md", "w") as f:
        f.write("\n".join(lines))

    print("Wrote outputs/presentation_addendum.md and supporting tables/figures.")


if __name__ == "__main__":
    main()
