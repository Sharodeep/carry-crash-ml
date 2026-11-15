"""Carry Trade Screener — Streamlit app. UI, filtering, and display only."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import ALL_CURRENCIES, EM_CURRENCIES, EM_STALE_DAYS, FIGURES_DIR, G10_CURRENCIES, TABLES_DIR
from data.rates import FredKeyMissingError, build_rate_table, em_fallback_is_stale, get_fred_client
from data.screener import build_screener_table
from research.live_score import load_cached_reading, load_regime_history, models_available, score_latest

st.set_page_config(
    page_title="Carry Trade Screener",
    page_icon="\U0001f4b1",
    layout="wide",
)

CUSTOM_CSS = """
<style>
:root {
    --strong-color: #00c896;
    --moderate-color: #f5a623;
    --weak-color: #e05c5c;
}
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}
.stMetric, [data-testid="stMetricValue"] {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
[data-testid="stDataFrame"] {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Section 1: Header ---
st.title("Carry Trade Screener")
st.caption(
    "Ranking G10 + EM pairs by carry adjusted for volatility, with a machine learning "
    "carry crash warning"
)

# --- FRED client / key check ---
try:
    fred = get_fred_client()
except FredKeyMissingError as exc:
    st.error(str(exc))
    st.stop()


# --- Section 2: Carry regime panel ---
@st.cache_data(ttl=86400)
def _get_crash_reading():
    try:
        return score_latest()
    except Exception:
        return load_cached_reading()


@st.cache_data(ttl=86400)
def _get_regime_history():
    return load_regime_history()


REGIME_COLORS = {"Calm": "#00c896", "Elevated": "#f5a623", "High": "#e05c5c"}

if models_available():
    st.subheader("Carry Stress Monitor")
    reading = _get_crash_reading()
    regime = reading["regime"]
    badge_color = REGIME_COLORS.get(regime, "#888888")

    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:14px; margin-bottom:4px;">
            <span style="background-color:{badge_color}; color:#0e1117; font-weight:700;
                  padding:4px 14px; border-radius:4px; font-family:monospace; font-size:1.1em;">
                {regime.upper()}
            </span>
            <span style="font-family:monospace; font-size:1.05em;">
                (percentile {reading['percentile_vs_history']:.0f} vs. 18yr history) &middot;
                model score (not a calibrated probability): {reading['logit_prob']:.2f}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Like an earthquake forecast: this says conditions are building toward a "
        "shock, not that one is about to happen on a given week."
    )

    driver_phrases = []
    for driver in reading["top_drivers"]:
        label = driver["feature"].replace("_", " ")
        driver_phrases.append(f"{label} ({driver['direction']} risk)")
    st.caption("Top drivers this week: " + "; ".join(driver_phrases))

    regime_history = _get_regime_history()
    fig_regime = go.Figure()
    fig_regime.add_trace(
        go.Scatter(
            x=regime_history.index,
            y=regime_history["logit_prob"],
            mode="lines",
            line=dict(color="#888888", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for regime_name, color in REGIME_COLORS.items():
        mask = regime_history["regime"] == regime_name
        fig_regime.add_trace(
            go.Scatter(
                x=regime_history.index[mask],
                y=regime_history.loc[mask, "logit_prob"],
                mode="markers",
                marker=dict(color=color, size=5),
                name=regime_name,
            )
        )
    fig_regime.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Crash probability (logit)",
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig_regime, width="stretch")

    live_note = "live" if reading.get("live") else "cached (live scoring failed)"
    training_end = reading.get("training_end_date", "unknown")
    st.caption(
        f"Estimated on a G10 carry basket. As of {reading['as_of_week']} ({live_note}). "
        f"Model trained through {training_end}."
    )

    if regime == "High":
        st.warning(
            "Carry crash risk is elevated. Rankings below do not account for this."
        )
else:
    st.info(
        "Crash early-warning model not found — run `python -m research.run_all` and "
        "commit `models/` to enable the regime panel."
    )

st.divider()

# --- Section 3: Sidebar controls ---
with st.sidebar:
    st.header("Filters")
    min_carry = st.slider("Min carry (%)", 0.0, 10.0, 0.5, 0.1)
    st.caption("Raw carry for the G10 tab, adjusted carry for the EM tab.")
    max_vol = st.slider("Max volatility (ann. %)", 0, 50, 30, 1)
    show_currencies = st.multiselect(
        "Show pairs containing", ALL_CURRENCIES, default=ALL_CURRENCIES
    )
    signal_options = ["Strong", "Moderate", "Weak / Avoid"]
    signal_filter = st.multiselect(
        "Signal filter", signal_options, default=signal_options
    )
    if st.button("\U0001f504 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# --- Data pipeline ---
rate_df, rate_warnings = build_rate_table(fred)

static_rows = rate_df[rate_df["source"] == "Static"]
stale_static = static_rows[
    static_rows["last_updated"].apply(
        lambda d: em_fallback_is_stale(d, EM_STALE_DAYS)
    )
]
if not static_rows.empty and len(stale_static) == len(static_rows):
    st.warning(
        "FRED is unavailable for "
        f"{', '.join(stale_static['currency'])} and the emergency static "
        f"rate(s) are stale (> {EM_STALE_DAYS} days old). Update `config.py` "
        "before relying on this data."
    )

screener_df, vol_warnings = build_screener_table(rate_df)

st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Apply the currency-membership filter globally; everything else is
# tab-specific because G10 and EM rank on different carry/vol columns ---
base_df = screener_df[
    screener_df["long_ccy"].isin(show_currencies) | screener_df["short_ccy"].isin(show_currencies)
].copy()

SIGNAL_COLORS = {
    "Strong": "#00c896",
    "Moderate": "#f5a623",
    "Weak / Avoid": "#e05c5c",
}


def _style_signal(val):
    color = SIGNAL_COLORS.get(val, "")
    return f"color: {color}; font-weight: 600;" if color else ""


def _render_ranking_and_scatter(
    group_df: pd.DataFrame,
    *,
    carry_col: str,
    vol_col: str,
    ratio_col: str,
    signal_col: str,
    carry_label: str,
    vol_label: str,
    ratio_label: str,
    key_prefix: str,
):
    group_df = group_df.dropna(subset=[vol_col, ratio_col]).copy()
    group_df = group_df[
        (group_df[carry_col] >= min_carry)
        & (group_df[vol_col] <= max_vol)
        & (group_df[signal_col].isin(signal_filter))
    ]
    group_df = group_df.sort_values(ratio_col, ascending=False).reset_index(drop=True)
    group_df["rank"] = range(1, len(group_df) + 1)

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pairs screened", len(group_df))
    col2.metric("Strong signal pairs", int((group_df[signal_col] == "Strong").sum()))

    if not group_df.empty:
        highest_carry_row = group_df.loc[group_df[carry_col].idxmax()]
        best_ratio_row = group_df.loc[group_df[ratio_col].idxmax()]
        col3.metric(
            f"Highest {carry_label.lower()} pair",
            highest_carry_row["pair"],
            f"{highest_carry_row[carry_col]:.2f}%",
        )
        col4.metric(
            f"Best {ratio_label} pair",
            best_ratio_row["pair"],
            f"{best_ratio_row[ratio_col]:.2f}",
        )
    else:
        col3.metric(f"Highest {carry_label.lower()} pair", "—")
        col4.metric(f"Best {ratio_label} pair", "—")

    # --- Main ranking table ---
    st.subheader("Ranking")

    display_df = group_df.copy()

    def _badge(row):
        marks = ""
        if row["stale_rate"]:
            marks += " ⚠️"
        if row["managed_rate"]:
            marks += " 🔒"
        return row["pair"] + marks

    display_df["pair"] = display_df.apply(_badge, axis=1)
    display_df = display_df.rename(
        columns={
            "rank": "Rank",
            "pair": "Pair",
            "long_rate": "Long Rate (%)",
            "short_rate": "Short Rate (%)",
            "carry_pct": "Carry (%)",
            "adj_carry_pct": "Adj Carry (%)",
            vol_col: vol_label,
            ratio_col: ratio_label,
            signal_col: "Signal",
        }
    )[
        [
            "Rank",
            "Pair",
            "Long Rate (%)",
            "Short Rate (%)",
            "Carry (%)",
            "Adj Carry (%)",
            vol_label,
            ratio_label,
            "Signal",
        ]
    ]

    styled = display_df.style.map(_style_signal, subset=["Signal"]).format(
        {
            "Long Rate (%)": "{:.2f}",
            "Short Rate (%)": "{:.2f}",
            "Carry (%)": "{:.2f}",
            "Adj Carry (%)": "{:.2f}",
            vol_label: "{:.2f}",
            ratio_label: "{:.2f}",
        }
    )
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "Pair": st.column_config.TextColumn(
                "Pair",
                help=(
                    "⚠️ = at least one leg's policy rate is the emergency static "
                    "fallback, not live FRED. 🔒 = managed currency (1yr vol vs. "
                    "USD < 3%): low daily volatility understates devaluation risk."
                ),
            )
        },
    )
    badge_notes = []
    if group_df["stale_rate"].any():
        badge_notes.append("⚠️ = static/stale rate fallback")
    if group_df["managed_rate"].any():
        badge_notes.append("🔒 = managed currency: low daily vol understates devaluation risk")
    if badge_notes:
        st.caption(" · ".join(badge_notes))

    # --- Carry vs Vol scatter plot ---
    st.subheader("Carry vs Volatility")

    if not group_df.empty:
        fig = px.scatter(
            group_df,
            x=vol_col,
            y=carry_col,
            size=group_df[ratio_col].abs(),
            color=signal_col,
            color_discrete_map=SIGNAL_COLORS,
            hover_name="pair",
            hover_data={
                "long_rate": ":.2f",
                "short_rate": ":.2f",
                "carry_pct": ":.2f",
                "adj_carry_pct": ":.2f",
                "vol_30d_pct": ":.2f",
                "vol_1y_pct": ":.2f",
                "vol_5y_pct": ":.2f",
                ratio_col: ":.2f",
            },
            labels={vol_col: vol_label, carry_col: carry_label, signal_col: "Signal"},
            template="plotly_dark",
        )
        fig.add_vline(x=max_vol, line_dash="dash", line_color="#888888")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_scatter")
    else:
        st.info("No pairs match the current filters.")


tab_g10, tab_em = st.tabs(["G10", "EM"])
with tab_g10:
    g10_df = base_df[
        base_df["long_ccy"].isin(G10_CURRENCIES) & base_df["short_ccy"].isin(G10_CURRENCIES)
    ]
    _render_ranking_and_scatter(
        g10_df,
        carry_col="carry_pct",
        vol_col="vol_g10_pct",
        ratio_col="carry_to_vol_g10",
        signal_col="signal_g10",
        carry_label="Carry",
        vol_label="Vol (ann. %, 30d/1y max)",
        ratio_label="Carry/Vol (raw)",
        key_prefix="g10",
    )
with tab_em:
    em_df = base_df[
        base_df["long_ccy"].isin(EM_CURRENCIES) | base_df["short_ccy"].isin(EM_CURRENCIES)
    ]
    _render_ranking_and_scatter(
        em_df,
        carry_col="adj_carry_pct",
        vol_col="vol_5y_pct",
        ratio_col="adj_carry_to_vol_em",
        signal_col="signal_em",
        carry_label="Adj Carry",
        vol_label="Vol (ann. %, 5y)",
        ratio_label="Adj Carry/Vol (5y vol)",
        key_prefix="em",
    )

# --- Section 6: Rate table ---
with st.expander("\U0001f4cb Underlying Rate Data"):
    rate_display = rate_df.copy()
    rate_display["badge"] = rate_display.apply(
        lambda row: "⚠️ Static rate" if row["source"] == "Static" else "",
        axis=1,
    )
    st.dataframe(
        rate_display.rename(
            columns={
                "currency": "Currency",
                "rate": "Rate (%)",
                "source": "Source",
                "last_updated": "Last Updated",
                "badge": "",
            }
        ),
        width="stretch",
        hide_index=True,
    )

# --- Section: Backtest evidence ---
if models_available():
    with st.expander("\U0001f4c8 Track record: does the crash warning work?"):
        backtest_path = f"{TABLES_DIR}/backtest_metrics.csv"
        try:
            backtest_df = pd.read_csv(backtest_path, index_col=0)
            uncond_sharpe = backtest_df.loc["unconditional", "sharpe"]
            vol_sharpe = backtest_df.loc["vol_targeted", "sharpe"]
            logit_sharpe = backtest_df.loc["logit_signal", "sharpe"]
            st.markdown(
                f"""
Walk-forward backtest, 2008 onward, net of 5bp transaction costs. The
**logit-signal strategy** (duck out of the carry basket when the model's
crash probability is in its top 20%) had the best risk-adjusted return:
Sharpe **{logit_sharpe:.2f}**, vs. **{uncond_sharpe:.2f}** for unconditional
carry and **{vol_sharpe:.2f}** for a plain volatility-targeting benchmark —
reported as-is, including that vol-targeting underperformed unconditional
carry in this sample.
"""
            )
            st.dataframe(backtest_df.round(4), width="stretch")
        except FileNotFoundError:
            st.info("Run `python -m research.run_all` to generate backtest results.")

        col_a, col_b = st.columns(2)
        try:
            col_a.image(f"{FIGURES_DIR}/04_backtest_cumulative.png", width="stretch")
            col_b.image(f"{FIGURES_DIR}/05_drawdowns.png", width="stretch")
        except Exception:
            pass

# --- Section 7: Methodology ---
with st.expander("ℹ️ Methodology"):
    st.markdown(
        """
**Carry trade**: going long a high-yielding currency and short a low-yielding one,
earning the interest rate differential ("carry") as long as the exchange rate
doesn't move against you enough to erase it.

**Carry/Vol**: the FX equivalent of a Sharpe ratio for carry, but computed
differently per tab. **G10** ranks on raw `carry_pct / vol_g10_pct` (vol_g10
is `max(30-day, ~1-year annualised vol)`, so a calm recent month can't hide
longer-run risk) — adjusted carry is shown alongside as an info column only,
it doesn't affect G10 ranking or signal. **EM** ranks on
`adj_carry_pct / vol_5y_pct`, a ~5-year annualised vol, so a calm recent
half-decade can't hide an older step-devaluation.

**Why "adjusted" carry**: a currency on a steady, managed depreciation path
(e.g. TRY) can look artificially attractive on raw carry/vol — its realised
volatility is low precisely because the move is slow and steady, even though
that steady move is quietly eating the carry. `adj_carry_pct = carry_pct -
max(trailing 12-month depreciation, 0)` — it only ever subtracts
depreciation of the long currency vs. the short currency; an *appreciating*
currency never gets a carry boost from this adjustment. Currencies whose
1-year vol vs. USD is under 3% are flagged 🔒 "managed" directly in the
ranking table, since their low day-to-day volatility is itself a sign this
adjustment matters for them. Pairs are also split into G10 and EM tabs so a
few very-high-carry EM pairs can't crowd G10 pairs out of view.

**Data sources**: policy rates come from FRED (mostly OECD "immediate rate /
call money" series, a monthly proxy for each central bank's short-term rate,
not necessarily the literal policy rate). Spot FX and realised volatility
come from yfinance daily closes. Both are cached (rates for 1 hour, FX for 15
minutes) to stay within API rate limits.

**Emergency static rates**: all EM currencies shown are fetched live from
FRED, same as G10. `config.py` also holds a hardcoded emergency rate per EM
currency, used only if a FRED fetch ever fails with no cached prior value
(e.g. a series gets discontinued). If that path is ever used, the affected
row is flagged with a ⚠️ badge directly in the ranking table (not just the
expander below) and, if stale (> 60 days), a warning banner.

**Carry crash warning**: a logistic regression model trained on 26 years of
weekly G10-only data, predicting the probability of a G10 carry-basket crash
(bottom-decile 4-week forward return) 4 weeks out. Walk-forward tested with
an expanding window and a 4-week embargo, so every out-of-sample prediction
only ever used information available at the time. A gradient-boosted tree
was also tried; logistic regression was kept as the primary model because it
discriminates crash weeks from calm weeks more reliably out-of-sample (see
the backtest evidence above). EM currencies are excluded from the crash model
— they're a late addition to the live screener and don't have the same depth
of clean historical data the model was trained and tested on.
"""
    )

# --- Warnings ---
all_warnings = rate_warnings + vol_warnings
if all_warnings:
    with st.expander(f"⚠️ {len(all_warnings)} data warning(s)"):
        for warning in all_warnings:
            st.warning(warning)
