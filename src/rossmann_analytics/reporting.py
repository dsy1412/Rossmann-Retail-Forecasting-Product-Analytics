from __future__ import annotations

from pathlib import Path

import pandas as pd


def _number(value: float) -> str:
    return f"{value:,.3f}"


def _percent(value: float) -> str:
    return f"{value * 100:,.1f}%"


def _comparison_markdown(comparison: pd.DataFrame) -> str:
    columns = ["model", "mae", "rmse", "rmspe"]
    lines = [
        "| Model | MAE | RMSE | RMSPE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison[columns].itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.mae:.4f} | {row.rmse:.4f} | {row.rmspe:.4f} |"
        )
    return "\n".join(lines)


def write_model_report(
    path: Path,
    comparison: pd.DataFrame,
    segments: pd.DataFrame,
    promotion_lift: pd.DataFrame,
    root_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    cutoff: pd.Timestamp,
    validation_end: pd.Timestamp,
    data_checks: dict[str, object],
) -> None:
    best = comparison.iloc[0]
    baseline_rows = comparison.loc[comparison["model"] == "historical_average"]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else best
    rmspe_improvement = (
        (baseline["rmspe"] - best["rmspe"]) / baseline["rmspe"]
        if baseline["rmspe"] > 0
        else 0.0
    )

    worst_segments = segments.loc[segments["observations"] >= 100].nlargest(5, "rmspe")
    promo_overall = promotion_lift.loc[promotion_lift["segment"] == "overall"]
    promo_text = "Unavailable"
    if not promo_overall.empty:
        promo_text = _percent(float(promo_overall.iloc[0]["mean_observational_lift"]))

    top_features = feature_importance.head(10)
    feature_lines = (
        [f"- `{row.feature}`: {_percent(row.importance)} of normalized importance" for row in top_features.itertuples()]
        if not top_features.empty
        else ["- Feature importance was unavailable for the selected model."]
    )
    segment_lines = (
        [
            f"- `{row.segment} = {row.value}`: RMSPE {_number(row.rmspe)} "
            f"across {int(row.observations):,} store-days"
            for row in worst_segments.itertuples()
        ]
        if not worst_segments.empty
        else ["- No segment met the minimum sample threshold."]
    )
    root_lines = (
        [
            f"- {row.event_type.title()}: {int(row.events):,} flagged events, "
            f"median deviation {_percent(row.median_deviation)}, "
            f"promotion share {_percent(row.promo_share)}"
            for row in root_summary.itertuples()
        ]
        if not root_summary.empty
        else ["- No extreme events were available for summary."]
    )

    comparison_table = _comparison_markdown(comparison)
    report = f"""# Rossmann Model & Product Analytics Report

Generated from the local pipeline. Every number below is computed from the supplied Kaggle files.

## Executive summary

- Best validation model: **{best['model']}**
- Validation window: **{cutoff.date()} through {validation_end.date()}**
- Best MAE: **{_number(best['mae'])}**
- Best RMSE: **{_number(best['rmse'])}**
- Best RMSPE: **{_number(best['rmspe'])}**
- RMSPE improvement over the historical-average baseline: **{_percent(rmspe_improvement)}**
- Descriptive mean promotion lift versus each store's non-promotion average: **{promo_text}**

## Data audit

- Rows: **{int(data_checks['rows']):,}**
- Stores: **{int(data_checks['stores']):,}**
- Date range: **{data_checks['start_date']} to {data_checks['end_date']}**
- Duplicate store-date keys: **{int(data_checks['duplicate_store_dates']):,}**
- Negative-sales rows: **{int(data_checks['negative_sales_rows']):,}**
- Closed-store rows with positive sales: **{int(data_checks['closed_store_positive_sales_rows']):,}**

## Model comparison

{comparison_table}

## Segment guardrails

These are the five highest-error cohorts with at least 100 validation observations. They are investigation targets, not automatically model failures.

{chr(10).join(segment_lines)}

## Promotion analysis

Promotion lift is descriptive, not causal. Promotion timing may reflect expected demand, seasonality, and store strategy. The table in `reports/tables/promotion_lift.csv` should be used to define hypotheses and candidate experiment segments.

A credible next step is a store-level randomized test or a matched-control design. Use incremental sales or gross profit as the primary metric and monitor margin, stockouts, cannibalization, and post-promotion demand as guardrails.

## Root-cause investigation

Store-days in the top and bottom one percent of deviation from their trailing 28-day store baseline were flagged.

{chr(10).join(root_lines)}

The detailed event table preserves promotion, holiday, store-type, assortment, and competition context so an analyst can move from detection to a testable explanation.

## Leading model drivers

{chr(10).join(feature_lines)}

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
"""
    path.write_text(report, encoding="utf-8")

