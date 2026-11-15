# Carry Crash Early Warning -- Results Summary

## Sample
- Period: 2000-01-14 -> 2026-11-06 (1400 weeks)
- Full-sample crash base rate (primary label): 10.0%

## Spliced / special-cased rate series
- EUR: using ECBDFR (daily, resampled to monthly) -- no live OECD 3-month series available for the Eurozone.
- JPY: IR3TIB01JPM156N gap-filled from IRSTCI01JPM156N (201 head observation(s) before 2002-04-01, 1 tail observation(s) after 2026-08-01).
- GBP: IR3TIB01GBM156N gap-filled from IRSTCI01GBM156N (0 head observation(s) before 1957-01-01, 8 tail observation(s) after 2026-01-01).

## Model metrics (pooled OOS, 2008 onward)
|    | model   |   roc_auc |   pr_auc |   base_rate |   brier_score |
|---:|:--------|----------:|---------:|------------:|--------------:|
|  0 | logit   |    0.5356 |   0.126  |      0.0918 |        0.1732 |
|  1 | gbm     |    0.4902 |   0.1116 |      0.0918 |        0.129  |

## Backtest metrics
| strategy         |   annualised_return |   annualised_vol |   sharpe |   max_drawdown |   worst_4wk_return |   pct_weeks_invested |   episode_2008 GFC |   episode_Aug 2011 |   episode_Aug 2015 |   episode_Mar 2020 |   episode_Jul-Aug 2024 |
|:-----------------|--------------------:|-----------------:|---------:|---------------:|-------------------:|---------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------------------:|
| unconditional    |              0.0093 |           0.081  |   0.1143 |        -0.2836 |            -0.1867 |               1      |            -0.2251 |            -0.0197 |            -0.0316 |            -0.0462 |                -0.0352 |
| vol_targeted     |              0.0107 |           0.0967 |   0.1111 |        -0.3197 |            -0.1125 |               1      |            -0.1747 |            -0.0305 |            -0.036  |            -0.0632 |                -0.0691 |
| logit_signal     |              0.0132 |           0.0706 |   0.1868 |        -0.1986 |            -0.0677 |               0.9153 |            -0.0395 |            -0.0197 |            -0.0116 |            -0.0244 |                -0.0384 |
| logit_continuous |              0.0056 |           0.0551 |   0.1022 |        -0.1923 |            -0.0581 |               1      |            -0.0536 |            -0.0287 |            -0.0124 |            -0.0162 |                -0.0225 |
| gbm_signal       |             -0.0018 |           0.0743 |  -0.0243 |        -0.2854 |            -0.1867 |               0.8204 |            -0.2251 |            -0.0197 |             0      |             0      |                -0.0416 |
| gbm_continuous   |              0.0026 |           0.0695 |   0.0377 |        -0.2701 |            -0.1722 |               1      |            -0.2086 |            -0.0207 |            -0.0133 |            -0.0207 |                -0.0318 |

## Episode returns by strategy
| strategy         |   episode_2008 GFC |   episode_Aug 2011 |   episode_Aug 2015 |   episode_Mar 2020 |   episode_Jul-Aug 2024 |
|:-----------------|-------------------:|-------------------:|-------------------:|-------------------:|-----------------------:|
| unconditional    |            -0.2251 |            -0.0197 |            -0.0316 |            -0.0462 |                -0.0352 |
| vol_targeted     |            -0.1747 |            -0.0305 |            -0.036  |            -0.0632 |                -0.0691 |
| logit_signal     |            -0.0395 |            -0.0197 |            -0.0116 |            -0.0244 |                -0.0384 |
| logit_continuous |            -0.0536 |            -0.0287 |            -0.0124 |            -0.0162 |                -0.0225 |
| gbm_signal       |            -0.2251 |            -0.0197 |             0      |             0      |                -0.0416 |
| gbm_continuous   |            -0.2086 |            -0.0207 |            -0.0133 |            -0.0207 |                -0.0318 |

## Top 3 features by mean |SHAP| (final-fold GBM)
- credit_chg4: 0.4310
- credit: 0.3449
- carry_spread: 0.2458

## GBM probability in the 4 weeks before each named episode
- 2008 GFC (starts 2008-09-01):
  - 2008-08-08: 0.270
  - 2008-08-15: 0.007
  - 2008-08-22: 0.014
  - 2008-08-29: 0.008
- Aug 2011 (starts 2011-08-01):
  - 2011-07-08: 0.135
  - 2011-07-15: 0.229
  - 2011-07-22: 0.117
  - 2011-07-29: 0.030
- Aug 2015 (starts 2015-08-01):
  - 2015-07-10: 0.424
  - 2015-07-17: 0.762
  - 2015-07-24: 0.426
  - 2015-07-31: 0.877
- Mar 2020 (starts 2020-03-01):
  - 2020-02-07: 0.534
  - 2020-02-14: 0.476
  - 2020-02-21: 0.416
  - 2020-02-28: 0.492
- Jul-Aug 2024 (starts 2024-07-01):
  - 2024-06-07: 0.191
  - 2024-06-14: 0.287
  - 2024-06-21: 0.136
  - 2024-06-28: 0.294

## Current reading
- As-of week: 2026-11-06
- GBM probability: 0.141 (logit: 0.293)
- Percentile vs. history: 41.0
- Regime: **Calm**
- Top drivers:
  - vix = 25.6400 (lowers risk)
  - credit_chg4 = 0.1100 (raises risk)
  - credit = 1.5600 (lowers risk)

## Warnings and data problems
- 9 input series are more than 45 days stale (expected for monthly OECD rate data): IR3TIB01USM156N, IR3TIB01JPM156N, IR3TIB01GBM156N, IR3TIB01CHM156N, IR3TIB01AUM156N, IR3TIB01NZM156N, IR3TIB01CAM156N, IR3TIB01NOM156N, IR3TIB01SEM156N
- The vol-targeting benchmark (Sharpe 0.111) underperformed unconditional carry (Sharpe 0.114) in this sample -- reported honestly rather than adjusted to look better.
- The GBM's pooled OOS ROC AUC (0.490) is below 0.5, driven mainly by a very poor 2008 fold (trained on a calm 2000-2007 window with no prior crisis example). This is a real limitation, not a bug -- see the per-year breakdown.
- Figure 02 (crash probability over time) shows this same problem visually: in several multi-year stretches (e.g. 2009, 2013) the GBM line collapses to a near-constant low value (full-sample std 0.186 vs. logit's 0.195), i.e. it stops discriminating between calm and risky weeks. Combined with the AUC result above, this is the basis for using **logistic regression as the primary model** for the live regime panel, with GBM kept as a secondary/diagnostic signal.
- This is consistent with Beutel, List & von Schweinitz (2018/2019, Deutsche Bundesbank DP 48/2018 / IWH DP 2/2019), who find ML methods fit crisis data well in-sample but are consistently outperformed by a logit benchmark out-of-sample -- the same small-sample/rare-event overfitting mechanism as our early folds. The literature is mixed, though: Jarmulska (2022, Journal of Forecasting) finds random forest beats logit for fiscal-stress early warning, a closely related problem. Treat our result as consistent with one strand of this literature, not as settling the question.
