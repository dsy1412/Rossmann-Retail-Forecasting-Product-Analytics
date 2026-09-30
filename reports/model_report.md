# Rossmann Model & Product Analytics Report

Generated from the local pipeline. Every number below is computed from the supplied Kaggle files.

## Executive summary

- Best validation model: **lightgbm**
- Validation window: **2015-06-20 through 2015-07-31**
- Best MAE: **480.752**
- Best RMSE: **754.883**
- Best RMSPE: **0.113**
- RMSPE improvement over the historical-average baseline: **52.1%**
- Descriptive mean promotion lift versus each store's non-promotion average: **41.7%**

## Data audit

- Rows: **1,017,209**
- Stores: **1,115**
- Date range: **2013-01-01 to 2015-07-31**
- Duplicate store-date keys: **0**
- Negative-sales rows: **0**
- Closed-store rows with positive sales: **0**

## Model comparison

| Model | MAE | RMSE | RMSPE |
| --- | ---: | ---: | ---: |
| lightgbm | 480.7524 | 754.8832 | 0.1129 |
| ridge | 658.2792 | 1065.0008 | 0.1732 |
| historical_average | 1078.8512 | 1543.9994 | 0.2356 |

## Segment guardrails

These are the five highest-error cohorts with at least 100 validation observations. They are investigation targets, not automatically model failures.

- `competition_band = close`: RMSPE 0.124 across 11,844 store-days
- `is_any_holiday = 0`: RMSPE 0.118 across 33,481 store-days
- `StoreType = a`: RMSPE 0.117 across 25,284 store-days
- `Promo = 0`: RMSPE 0.115 across 30,105 store-days
- `competition_band = very_close`: RMSPE 0.114 across 9,240 store-days

## Promotion analysis

Promotion lift is descriptive, not causal. Promotion timing may reflect expected demand, seasonality, and store strategy. The table in `reports/tables/promotion_lift.csv` should be used to define hypotheses and candidate experiment segments.

A credible next step is a store-level randomized test or a matched-control design. Use incremental sales or gross profit as the primary metric and monitor margin, stockouts, cannibalization, and post-promotion demand as guardrails.

## Root-cause investigation

Store-days in the top and bottom one percent of deviation from their trailing 28-day store baseline were flagged.

- Drop: 100 flagged events, median deviation -95.4%, promotion share 21.0%
- Spike: 100 flagged events, median deviation 785.1%, promotion share 53.0%

The detailed event table preserves promotion, holiday, store-type, assortment, and competition context so an analyst can move from detection to a testable explanation.

## Leading model drivers

- `numeric__sales_lag_14`: 20.1% of normalized importance
- `numeric__Promo`: 14.6% of normalized importance
- `numeric__sales_lag_1`: 9.7% of normalized importance
- `numeric__sales_roll_mean_28`: 7.9% of normalized importance
- `numeric__sales_lag_7`: 7.9% of normalized importance
- `numeric__sales_lag_28`: 7.5% of normalized importance
- `numeric__sales_roll_mean_14`: 5.0% of normalized importance
- `numeric__day`: 4.0% of normalized importance
- `numeric__sales_roll_mean_7`: 3.4% of normalized importance
- `numeric__DayOfWeek`: 3.2% of normalized importance

Feature importance describes predictive reliance, not causal impact. Correlated calendar and lag variables can split or share importance.

## Product recommendations

1. Prioritize model iteration on the largest high-error segment rather than optimizing only aggregate RMSPE.
2. Use promotion-lift heterogeneity to choose experiment strata, then estimate incremental impact with a randomized or quasi-experimental design.
3. Turn spike/drop alerts into a recurring review that joins inventory, pricing, and local-event data before assigning a cause.
4. Monitor forecast error separately during promotions and holidays because those are high-decision-value periods.

## Limitations

- Validation represents rolling one-day-ahead forecasts: lag features use sales observed before each predicted day.
- Price, inventory, margin, customer, and local-event variables are unavailable.
- Promotion effects are observational and should not be presented as causal.
- The dataset is historical and demonstrates methodology rather than current market conditions.
