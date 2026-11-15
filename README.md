# Carry Trade Screener

A research tool for FX carry trades. It screens G10 + select EM currency pairs by interest
rate differential, overlays realised volatility, and ranks every pair by carry-to-vol — the
FX equivalent of a Sharpe ratio for carry. G10 and EM get separate tabs with separate
ranking logic (see [Methodology](#methodology)), because a currency on a managed
depreciation path needs different treatment than a freely floating one. It also runs a
**Carry Stress Monitor**: a logistic regression trained on 26 years of weekly G10 data,
predicting the probability of a carry-basket crash 4 weeks out, walk-forward backtested and
shown live as a regime panel at the top of the app. Built with Streamlit, FRED, yfinance,
and scikit-learn.

[Live on Hugging Face Spaces](#)

![Screenshot placeholder](docs/screenshot.png)

## How to run locally

```bash
git clone <this-repo>
cd carry-trade-screener
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add your FRED API key (free at [fred.stlouisfed.org](https://fred.stlouisfed.org)):

```bash
# For the Streamlit app
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
[fred]
api_key = "YOUR_FRED_API_KEY_HERE"
EOF

# For the research pipeline (separate from the app, uses .env instead)
echo "FRED_API_KEY=YOUR_FRED_API_KEY_HERE" > .env
```

Run the screener:

```bash
streamlit run app.py
```

The crash model that powers the regime panel is trained offline, not inside the app. To
(re)train it and regenerate every figure/table the app and this README reference:

```bash
python -m research.run_all
```

This writes `models/*.joblib`, `outputs/current_reading.json`, `outputs/results_summary.md`,
and `outputs/figures/*.png`. Commit the outputs — the app loads the committed model files
and only re-scores the latest week live; it never retrains on load.

## Crash early-warning model (Carry Stress Monitor)

Each week, the model estimates the probability that the G10 carry basket (long the top-3 /
short the bottom-3 G10 currencies by policy rate) will crash — a bottom-decile 4-week
forward return — 4 weeks from now. It's trained on weekly data from 2000-01-14 to
2026-11-06 (1,400 weeks), with every feature and label built so that no information from
after week *t* ever leaks into a prediction made at week *t*, and evaluated with an
expanding walk-forward split (first test year 2008) plus a 4-week embargo between train and
test. The app's regime panel reports the model's output as a raw score, not a calibrated
probability, and frames it like an earthquake forecast: a reading of elevated stress means
conditions are building toward a shock, not that one is imminent on a specific week.

**Headline backtest result**: a strategy that exits the carry basket whenever the model's
crash probability is in its top 20% had a Sharpe ratio of **0.19** over 2008–2026 (net of
transaction costs), versus **0.11** for holding the basket unconditionally and **0.11** for
a plain volatility-targeting benchmark. During the 2008 financial crisis specifically, this
cut the drawdown from −22.5% (unconditional) to −3.95%. Most of that edge over
unconditional carry is concentrated in 2008 — with the GFC excluded, the gap narrows
substantially (see `outputs/presentation_addendum.md`).

Two models were tried: logistic regression and a gradient-boosted tree. **Logistic
regression is used as the primary model** — the tree model's pooled out-of-sample ROC AUC
(0.49) came in below 0.5 (worse than random) and visibly flat-lines across much of the
sample, driven by overfitting in small-sample early folds, a known failure mode in this
literature. This choice has some support in the literature on crisis early-warning models
(logit often matches or beats ML on small, low-base-rate samples like this one — e.g. Beutel,
List & von Schweinitz, 2018/2019), though the evidence is mixed overall (e.g. Jarmulska,
2022, finds a random forest beats logit for fiscal stress) — see
`outputs/results_summary.md` for the full writeup and citations. This is reported as a real
limitation, not smoothed over.

EM currencies are excluded from the crash model — they weren't part of the original 26-year
training sample and don't have the same depth of clean historical rate data.

## Methodology

**Carry trade**: going long a high-yielding currency and short a low-yielding one, earning the interest rate differential ("carry") as long as the exchange rate doesn't move against you enough to erase it.

**Two ranking metrics, one per tab** — a currency on a managed depreciation path (classically, TRY) can look like a "Strong" carry trade on raw carry/vol alone, because its low day-to-day volatility hides a steady devaluation that keeps eating the carry. So G10 and EM pairs are ranked differently:
- **G10 tab**: ranks on raw `carry_pct / vol_g10_pct`, where `vol_g10_pct` is the max of 30-day and ~1-year annualised volatility. This is the classic FX Sharpe-ratio-for-carry.
- **EM tab**: ranks on `adj_carry_pct / vol_5y_pct` — adjusted carry over a 5-year annualised volatility window, so a calm recent half-decade can't hide an older step-devaluation. `adj_carry_pct = carry_pct - max(trailing 12-month depreciation, 0)`: it only ever subtracts depreciation of the long currency vs. the short currency over the trailing year — appreciation never adds to carry. Raw carry is still shown as an info column.

Either way, a pair rewards a wide rate differential *and* low realised volatility, and penalises pairs where a large carry is offset by a choppy (or, for EM, steadily depreciating) exchange rate. Any currency whose 1-year volatility vs. USD is under 3% gets a 🔒 "managed currency" badge in both tabs, flagging that its low observed volatility likely understates devaluation risk.

Cross-pair prices and volatility are derived by triangulation through USD (`LONG/SHORT = (LONG/USD) / (SHORT/USD)`) rather than fetched per-pair. This is exact, not approximate — `log-return(LONG/SHORT) = log-return(LONG/USD) - log-return(SHORT/USD)` by construction — and it's how the market itself prices illiquid crosses that have no direct quote. It also means one batched download covers every pair instead of ~110 individual (and often missing) ticker lookups.

**Data sources and limitations**: policy rates come from [FRED](https://fred.stlouisfed.org), mostly OECD "immediate rate / call money" or 3-month interbank series — a reasonable monthly proxy for each central bank's short-term rate, not necessarily the literal policy rate. Spot FX and realised volatility come from [yfinance](https://github.com/ranaroussi/yfinance) daily closes. Both are cached (rates for 1 hour, FX for 15 minutes).

**Emergency static rates**: BRL, ZAR, INR, TRY, and IDR are all fetched live from FRED, same as G10. `config.py` also holds a hardcoded emergency rate per EM currency, used only if a FRED fetch ever fails with no cached prior value (e.g. a series gets discontinued). If that path is ever used, the affected row is flagged with a badge in the rate table and, if stale (> 60 days), a warning banner.

**Crash model data and limitations**: trained on the same FRED rate/FX series as the live screener plus VIX, BAA-10Y credit spread, and Brent crude (all FRED). Policy rates are lagged one month to respect real-world publication delay — no feature ever uses a rate before it was actually public. Interpretability comes from coefficient-based local explanations for logit (scikit-learn) and SHAP (`shap` library) for the gradient-boosted tree. The sample contains only a handful of true crisis episodes (2008, Aug 2011, Aug 2015, Mar 2020, Jul–Aug 2024), and the 4-week-forward labels overlap week to week, so the effective independent sample size behind these results is much smaller than the 1,386 weeks of data suggests — treat the backtest as indicative, not as a guarantee of future performance.

## Author

[Your Name](https://github.com/yourhandle)
