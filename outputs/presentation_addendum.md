# Presentation addendum

## 1. Backtest excluding the 2008 GFC (2009-01-01 onward)
| strategy         |   annualised_return |   annualised_vol |   sharpe |   max_drawdown |   worst_4wk_return |   pct_weeks_invested |   episode_2008 GFC |   episode_Aug 2011 |   episode_Aug 2015 |   episode_Mar 2020 |   episode_Jul-Aug 2024 |
|:-----------------|--------------------:|-----------------:|---------:|---------------:|-------------------:|---------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------------------:|
| unconditional    |              0.0243 |           0.0714 |   0.3401 |        -0.1987 |            -0.0745 |               1      |                nan |            -0.0197 |            -0.0316 |            -0.0462 |                -0.0352 |
| vol_targeted     |              0.0239 |           0.0952 |   0.2513 |        -0.3197 |            -0.1125 |               1      |                nan |            -0.0305 |            -0.036  |            -0.0632 |                -0.0691 |
| logit_signal     |              0.0188 |           0.0672 |   0.2802 |        -0.1986 |            -0.0606 |               0.9159 |                nan |            -0.0197 |            -0.0116 |            -0.0244 |                -0.0384 |
| logit_continuous |              0.0105 |           0.0519 |   0.2026 |        -0.1923 |            -0.0528 |               1      |                nan |            -0.0287 |            -0.0124 |            -0.0162 |                -0.0225 |
| gbm_signal       |              0.0119 |           0.064  |   0.1852 |        -0.256  |            -0.0606 |               0.8211 |                nan |            -0.0197 |             0      |             0      |                -0.0416 |
| gbm_continuous   |              0.0161 |           0.06   |   0.2683 |        -0.2103 |            -0.0602 |               1      |                nan |            -0.0207 |            -0.0133 |            -0.0207 |                -0.0318 |
_`episode_2008 GFC` is NaN by construction: that window falls entirely before the 2009-01-01 cutoff, so there are no weeks left to sum._

Plain language: with the 2008 crisis removed, unconditional carry's Sharpe is 0.340 and the logit-signal strategy's is 0.280. Most of the logit signal's advantage over unconditional carry comes from the 2008 episode specifically -- worth saying explicitly rather than letting the headline Sharpe imply broader, steadier outperformance.

## 2. Logit signal: out-of-market spells
| start      | end        |   n_weeks |   unconditional_basket_return |
|:-----------|:-----------|----------:|------------------------------:|
| 2008-01-25 | 2008-01-25 |         1 |                        0.0131 |
| 2008-10-03 | 2008-10-24 |         4 |                       -0.1867 |
| 2011-09-02 | 2011-09-02 |         1 |                       -0.0018 |
| 2012-06-15 | 2012-06-15 |         1 |                        0.0136 |
| 2014-02-14 | 2014-02-14 |         1 |                        0.0075 |
| 2014-05-23 | 2014-05-30 |         2 |                       -0.0071 |
| 2014-08-15 | 2014-08-22 |         2 |                        0.0102 |
| 2014-09-05 | 2014-09-05 |         1 |                        0.0045 |
| 2014-10-17 | 2015-02-27 |        20 |                       -0.0199 |
| 2015-03-20 | 2015-04-03 |         3 |                       -0.0129 |
| 2015-05-15 | 2015-05-22 |         2 |                       -0.0004 |
| 2015-07-03 | 2015-08-21 |         8 |                       -0.0297 |
| 2015-10-02 | 2015-10-23 |         4 |                        0.045  |
| 2016-01-01 | 2016-01-01 |         1 |                        0.0044 |
| 2016-01-22 | 2016-02-19 |         5 |                        0.0218 |
| 2017-04-28 | 2017-04-28 |         1 |                       -0.0234 |
| 2018-03-09 | 2018-03-16 |         2 |                        0.0031 |
| 2018-11-09 | 2018-12-07 |         5 |                        0.0071 |
| 2019-01-04 | 2019-01-18 |         3 |                        0.0185 |
| 2019-06-14 | 2019-06-14 |         1 |                       -0.0001 |
| 2020-03-13 | 2020-04-10 |         5 |                        0.0056 |
| 2020-05-15 | 2020-05-15 |         1 |                       -0.0042 |
| 2023-03-17 | 2023-03-31 |         3 |                        0.0001 |
| 2023-05-05 | 2023-05-19 |         3 |                        0.0195 |
| 2024-01-12 | 2024-01-12 |         1 |                        0.0026 |
| 2024-06-28 | 2024-07-05 |         2 |                        0.0096 |

The logit-signal strategy had **zero** out-of-market weeks in 2009 (fully invested all 52 weeks) -- the 'return avoided' question doesn't apply that year. The strategy rode the 2009 carry recovery unconditionally, same as the unconditional benchmark.

## 3. Logit driver figures (replacing GBM SHAP as headline)
`outputs/figures/06b_logit_drivers.png` -- standardized coefficients, final-fold logit model, sorted by absolute size:
|                 |   standardized_coefficient |
|:----------------|---------------------------:|
| credit_chg4     |                     0.5879 |
| vix             |                    -0.4999 |
| carry_spread    |                     0.2181 |
| basket_vol12    |                     0.2071 |
| basket_mom4     |                     0.2056 |
| credit          |                     0.164  |
| basket_mom12    |                    -0.1179 |
| basket_vol_chg4 |                     0.1105 |
| brent_ret12     |                    -0.0836 |
| jpy_ret4        |                    -0.0675 |
| vix_chg4        |                    -0.0412 |

`outputs/figures/07b_logit_aug2024.png` -- coefficient x standardized-value contributions for the week of 2024-08-02 (before the yen carry unwind), using the 2024 walk-forward fold's logit model:
| label                            |   value |    shap | direction   |
|:---------------------------------|--------:|--------:|:------------|
| Credit spread 4-week change      |  0.24   |  0.6969 | raises      |
| Basket 4-week momentum           | -0.0478 | -0.4715 | lowers      |
| JPY 4-week return                |  0.0894 | -0.3028 | lowers      |
| VIX level                        | 23.39   | -0.213  | lowers      |
| Credit spread (BAA10Y) level     |  1.8    | -0.1494 | lowers      |
| Basket 12-week momentum          | -0.0324 |  0.1156 | raises      |
| Basket volatility 4-week change  |  0.0253 |  0.096  | raises      |
| VIX 4-week change                | 10.91   | -0.0937 | lowers      |
| Brent crude 12-week return       | -0.0623 |  0.0511 | raises      |
| Carry spread (long - short rate) |  3.6002 |  0.0437 | raises      |
| Basket 12-week volatility        |  0.0725 |  0.0025 | raises      |

## 4. Current reading percentile: logit or GBM?
- `outputs/current_reading.json` currently uses **logit** for `percentile_vs_history` (percentile = 41.0).
- Recomputed GBM-based percentile: 48.5; recomputed logit-based percentile: 41.0.
- Note: this project switched the live regime panel to logit as the primary model earlier (see `outputs/results_summary.md`'s rationale -- GBM's pooled OOS ROC AUC is below 0.5). `current_reading.json` already reflects that decision. `outputs/current_reading_logit.json` is written alongside it for this check; the two should already agree.

## 5. Logit readings in the 4 weeks before each crisis episode
| episode      | date       |   logit_prob |   percentile_vs_logit_oos_history | regime   |
|:-------------|:-----------|-------------:|----------------------------------:|:---------|
| 2008 GFC     | 2008-08-08 |       0.0392 |                               8   | Calm     |
| 2008 GFC     | 2008-08-15 |       0.0383 |                               7.8 | Calm     |
| 2008 GFC     | 2008-08-22 |       0.0489 |                               8.6 | Calm     |
| 2008 GFC     | 2008-08-29 |       0.0762 |                              11.7 | Calm     |
| Aug 2011     | 2011-07-08 |       0.2039 |                              28.1 | Calm     |
| Aug 2011     | 2011-07-15 |       0.1812 |                              25   | Calm     |
| Aug 2011     | 2011-07-22 |       0.2412 |                              33.7 | Calm     |
| Aug 2011     | 2011-07-29 |       0.1711 |                              23.8 | Calm     |
| Aug 2015     | 2015-07-10 |       0.6567 |                              94.8 | High     |
| Aug 2015     | 2015-07-17 |       0.6965 |                              95.7 | High     |
| Aug 2015     | 2015-07-24 |       0.6257 |                              93.6 | High     |
| Aug 2015     | 2015-07-31 |       0.737  |                              97.1 | High     |
| Mar 2020     | 2020-02-07 |       0.4263 |                              68.1 | Elevated |
| Mar 2020     | 2020-02-14 |       0.5108 |                              84.2 | Elevated |
| Mar 2020     | 2020-02-21 |       0.4575 |                              75.2 | Elevated |
| Mar 2020     | 2020-02-28 |       0.2018 |                              27.7 | Calm     |
| Jul-Aug 2024 | 2024-06-07 |       0.4803 |                              79.6 | Elevated |
| Jul-Aug 2024 | 2024-06-14 |       0.5076 |                              83.5 | Elevated |
| Jul-Aug 2024 | 2024-06-21 |       0.5241 |                              85.8 | Elevated |
| Jul-Aug 2024 | 2024-06-28 |       0.5816 |                              91.1 | High     |

Caveat: `percentile_vs_logit_oos_history` here is computed against the **full** pooled walk-forward OOS history (through 2026), for descriptive ranking only -- it is not a point-in-time percentile (a stricter version would only use OOS history available up to that fold). None of the backtest or model-selection results elsewhere use this full-history percentile; the actual signal-threshold logic in `research/backtest.py` only ever uses each fold's own training-window quantile.

## Anything surprising
- Most of the logit strategy's edge over unconditional carry is concentrated in 2008 -- ex-GFC, the gap narrows substantially (or disappears).
- The longest logit out-of-market spell ran 20 consecutive weeks -- worth checking that isn't an artifact of one fold's threshold before presenting it as responsive risk management.
- The 4-weeks-before-episode table (section 5) shows the model was NOT reliably in 'High' regime ahead of every crisis -- read the regime column fold by fold rather than assuming uniform early warning across all five episodes.
