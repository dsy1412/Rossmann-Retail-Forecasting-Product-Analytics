# Rossmann Retail Forecasting & Product Analytics

An end-to-end data science project that forecasts daily store sales and explains the business drivers behind changes in performance. The project is designed for Product Data Science and applied analytics recruiting: it combines reliable prediction with metric design, segment diagnostics, promotion analysis, root-cause investigation, and decision-oriented reporting.

## Business problem

Retail teams need to answer two related questions:

1. What sales should each store expect over the next several weeks?
2. Why are sales changing, and which levers deserve operational attention?

The project predicts daily sales at the store level and evaluates the effects associated with promotions, holidays, store type, assortment, competition distance, and seasonality. Promotion results are treated as observational associations rather than causal estimates because promotions were not randomly assigned.

## Dataset

The project uses the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/competitions/rossmann-store-sales/data):

- `train.csv`: daily store observations and sales
- `store.csv`: store characteristics and competition information
- `test.csv`: optional Kaggle test set for future submission work

Raw competition files are intentionally excluded from version control. See [data/README.md](data/README.md) for setup instructions.

## Repository structure

```text
rossmann-product-analytics/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/                  # Kaggle CSV files; ignored by Git
|   `-- processed/            # Cached feature tables; ignored by Git
|-- notebooks/
|   `-- 01_rossmann_product_analytics.ipynb
|-- scripts/
|   `-- run_pipeline.py
|-- src/rossmann_analytics/
|   |-- analytics.py          # Promotion lift and root-cause analysis
|   |-- config.py
|   |-- data.py
|   |-- features.py           # Leakage-safe feature engineering
|   |-- metrics.py
|   |-- modeling.py
|   |-- pipeline.py
|   |-- reporting.py
|   `-- visualization.py
|-- reports/
|   |-- figures/
|   `-- tables/
|-- outputs/                  # Metrics, predictions, and fitted model
`-- tests/
```

## Reproducible setup

```powershell
cd E:\master\upenn\rossmann-product-analytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the Kaggle competition files and place `train.csv` and `store.csv` in `data/raw/`. Then run:

```powershell
python scripts/run_pipeline.py
```

Useful options:

```powershell
# Faster development run using the most recent 25% of each store's history
python scripts/run_pipeline.py --sample-frac 0.25

# Change the final time holdout from 42 to 56 days
python scripts/run_pipeline.py --validation-days 56

# Skip SHAP when running on a constrained machine
python scripts/run_pipeline.py --no-shap
```

The command writes:

- EDA figures to `reports/figures/`
- performance and analytics tables to `reports/tables/`
- predictions, model metrics, and the best fitted model to `outputs/`
- an evidence-grounded summary to `reports/model_report.md`

## Methods

### Exploratory analysis

The pipeline audits missing values and data quality, then visualizes:

- sales distribution and monthly trend
- sales by store type and assortment
- sales by promotion status and day of week
- behavior around state and school holidays
- competition distance versus sales

### Leakage-safe feature engineering

Features include calendar variables, holiday indicators, promotion state, competition history, store characteristics, cyclical seasonality, store-level sales lags, rolling statistics, and prior promotion state. Every target-derived feature is shifted before rolling calculations, so the current day's sales never enter its own predictors.

`Customers` is excluded because it is unavailable at forecast time in the Kaggle test set and would create an unrealistic deployment dependency.

### Validation strategy

The final 42 calendar days are held out by default. Earlier dates are used for training. This simulates forecasting future store performance and avoids the optimism produced by a random split.

Three model tiers are compared:

1. Store-by-day-of-week historical average baseline
2. Regularized linear model
3. LightGBM gradient-boosted trees, with a scikit-learn random forest fallback

Models are evaluated with MAE, RMSE, and RMSPE. The best model is also audited by store type, promotion status, holiday period, assortment, and competition-distance band.

## Product analytics layer

The analysis goes beyond a single leaderboard score:

- **North-star forecast metric:** RMSPE, paired with MAE and RMSE to keep percentage and dollar-scale error visible
- **Guardrail diagnostics:** segment error tables reveal whether aggregate performance hides weak store or holiday cohorts
- **Promotion analysis:** within-store non-promotion baselines estimate observational lift, then expose heterogeneous results by store type and assortment
- **Root-cause analysis:** unusually high and low store-days are detected relative to trailing store baselines and summarized by promotion, holiday, store, and assortment context
- **Explainability:** feature importance and optional SHAP values identify which signals drive predictions
- **Decision translation:** the automated report turns computed outputs into recommendations and explicitly labels unsupported causal interpretations

## Experimentation thinking

Historical promotion lift is useful for hypothesis generation, but it is not proof that a promotion caused the observed difference. A production follow-up should use randomized geo/store tests when feasible, or a quasi-experimental design with matched controls and pre-period adjustment. Recommended experiment metrics include:

- Primary: incremental sales or gross profit per eligible store-day
- Guardrails: margin, stockouts, cannibalization, and post-promotion demand dip
- Heterogeneity: store type, assortment, competition band, and customer/store maturity

## Results

The full pipeline was run on 1,017,209 store-day observations covering 1,115 stores. The validation window contains the final 42 days, from June 20 through July 31, 2015, and represents rolling one-day-ahead forecasts.

| Model | MAE | RMSE | RMSPE |
| --- | ---: | ---: | ---: |
| LightGBM | **480.75** | **754.88** | **0.1129** |
| Ridge | 658.28 | 1,065.00 | 0.1732 |
| Historical average | 1,078.85 | 1,544.00 | 0.2356 |

LightGBM reduced RMSPE by **52.1%** relative to the historical-average baseline. Results are reproducible from `outputs/metrics.json`, `reports/tables/model_comparison.csv`, and `reports/model_report.md`.

## Business insights

The business findings are descriptive and intended to guide further investigation and experiment design:

- Promotion store-days averaged **41.7% higher sales** than each store's non-promotion mean. This is observational lift, not a causal estimate.
- Stores in the farthest competition-distance band showed the largest descriptive promotion lift at **46.9%**, compared with **39.5%** for the medium-distance band.
- The highest-error validation cohort was the close competition-distance segment with **0.1244 RMSPE** across 11,844 store-days.
- SHAP identified the 14-day sales lag, promotion status, 1-day sales lag, and 28-day rolling mean as the leading prediction drivers.
- Promotions appeared on **53% of the 100 most extreme sales spikes**, compared with **21% of the 100 most extreme drops**. This association does not establish causality.

## Limitations

- Promotion assignment is observational and likely confounded by seasonality, store strategy, and expected demand.
- The dataset does not include prices, margins, inventory, local events, or digital engagement, which limits product and profitability conclusions.
- Lag features use observed prior sales. Multi-step production forecasts would require recursive prediction or direct horizon-specific models.
- The data is historical; the project demonstrates transferable methods rather than current retail conditions.
- RMSPE can be unstable near zero, so the implementation excludes zero-actual rows from that metric and reports MAE/RMSE alongside it.

## Future improvements

- Add direct 7-, 14-, and 42-day horizon models and forecast uncertainty intervals.
- Estimate causal promotion effects with matched controls, difference-in-differences, or randomized store tests.
- Add price, inventory, margin, and local-event data to optimize business outcomes rather than sales alone.
- Track experiments with MLflow and monitor drift after deployment.
- Build a compact Streamlit decision tool for store and segment drill-down.

