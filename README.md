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

No performance numbers are claimed before the pipeline is run on the Kaggle files. After execution, this section should be updated from:

- `outputs/metrics.json`
- `reports/tables/model_comparison.csv`
- `reports/tables/segment_performance.csv`
- `reports/tables/promotion_lift.csv`
- `reports/model_report.md`

This keeps the repository honest: every number presented to a recruiter can be traced to a saved artifact.

## Business insights

The automated report will populate evidence-backed findings after execution. The analysis is designed to answer:

- Which store segments benefit most from promotions?
- Where does forecast error increase during holidays or promotion periods?
- Are sales spikes explained by promotions, holidays, seasonality, or store mix?
- Does competition distance have a stable relationship with sales after segmentation?
- Which features most influence the model, and where should analysts investigate next?

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

## GitHub project summary

Built a reproducible retail forecasting and product analytics pipeline on the Rossmann Store Sales dataset. The project compares time-aware baselines and LightGBM, audits performance across business segments, measures observational promotion lift, diagnoses sales spikes and drops, and generates an evidence-grounded model report for product and operations decisions.

## Resume bullets

Use the generated metrics to replace bracketed fields after running the full pipeline:

- Built a leakage-safe retail forecasting pipeline across 1,000+ stores using Python, pandas, and LightGBM, comparing seasonal, regularized linear, and boosted-tree models with time-based validation and improving RMSPE from **[baseline]** to **[best model]**.
- Designed a Product Data Science analytics layer to quantify observational promotion lift across store and assortment segments, diagnose sales spikes and drops, and translate model outputs into testable recommendations for promotion strategy.
- Developed segment-level model monitoring and SHAP-based driver analysis across promotion, holiday, store-type, and competition cohorts, identifying **[highest-risk segment]** and reducing its forecast error by **[X% after iteration]**.

