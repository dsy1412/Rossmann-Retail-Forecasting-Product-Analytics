from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .analytics import promotion_lift_analysis, root_cause_events, root_cause_summary
from .config import ProjectConfig
from .data import (
    data_quality_checks,
    data_quality_summary,
    load_raw_data,
    prepare_training_data,
)
from .features import build_features, model_feature_columns
from .metrics import segment_performance
from .modeling import (
    best_model_result,
    feature_importance_table,
    model_comparison,
    shap_values_for_tree_model,
    time_based_split,
    train_and_evaluate_models,
)
from .reporting import write_model_report
from .visualization import (
    save_eda_figures,
    save_feature_importance_figure,
    save_model_comparison_figure,
    save_shap_summary,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_pipeline(config: ProjectConfig | None = None) -> dict[str, Any]:
    """Run the complete forecasting and product analytics workflow."""

    config = config or ProjectConfig()
    config.ensure_directories()

    print("[1/7] Loading and validating Rossmann data")
    train_raw, store_raw = load_raw_data(config)
    merged = prepare_training_data(train_raw, store_raw, sample_frac=config.sample_frac)
    checks = data_quality_checks(merged)
    data_quality_summary(merged).to_csv(
        config.tables_dir / "data_quality_summary.csv", index=False
    )
    _write_json(config.outputs_dir / "data_quality_checks.json", checks)

    print("[2/7] Creating exploratory figures")
    save_eda_figures(merged, config.figures_dir)

    print("[3/7] Building leakage-safe time-series and business features")
    featured = build_features(merged)
    try:
        featured.to_parquet(config.processed_data_dir / "model_features.parquet", index=False)
    except ImportError:
        featured.to_pickle(config.processed_data_dir / "model_features.pkl")
    features, numeric_features, categorical_features = model_feature_columns(featured)

    print("[4/7] Training time-aware baseline, linear, and boosted-tree models")
    training, validation, cutoff = time_based_split(featured, config.validation_days)
    results = train_and_evaluate_models(
        training,
        validation,
        features,
        numeric_features,
        categorical_features,
        config.random_state,
    )
    comparison = model_comparison(results)
    comparison.to_csv(config.tables_dir / "model_comparison.csv", index=False)
    save_model_comparison_figure(comparison, config.figures_dir)

    best = best_model_result(results)
    prediction_columns = [
        "Date",
        "Store",
        "Sales",
        "Open",
        "Promo",
        "StoreType",
        "Assortment",
        "StateHoliday",
        "SchoolHoliday",
        "is_any_holiday",
        "competition_band",
    ]
    predictions = validation[prediction_columns].copy()
    predictions["prediction"] = best.predictions
    predictions["residual"] = predictions["Sales"] - predictions["prediction"]
    predictions.to_csv(config.outputs_dir / "validation_predictions.csv", index=False)

    print("[5/7] Auditing model performance across decision-relevant segments")
    segments = segment_performance(
        predictions,
        actual_column="Sales",
        prediction_column="prediction",
        segment_columns=[
            "StoreType",
            "Promo",
            "is_any_holiday",
            "StateHoliday",
            "Assortment",
            "competition_band",
        ],
    )
    segments.to_csv(config.tables_dir / "segment_performance.csv", index=False)

    print("[6/7] Running promotion and root-cause analyses")
    promotion_lift = promotion_lift_analysis(featured)
    promotion_lift.to_csv(config.tables_dir / "promotion_lift.csv", index=False)
    events = root_cause_events(featured)
    events.to_csv(config.tables_dir / "sales_spike_drop_events.csv", index=False)
    event_summary = root_cause_summary(events)
    event_summary.to_csv(config.tables_dir / "sales_spike_drop_summary.csv", index=False)

    importance = feature_importance_table(best)
    importance.to_csv(config.tables_dir / "feature_importance.csv", index=False)
    save_feature_importance_figure(importance, config.figures_dir)

    shap_status = "disabled"
    report_importance = importance
    if config.run_shap:
        try:
            shap_values, transformed, names = shap_values_for_tree_model(
                best,
                validation[features],
                random_state=config.random_state,
            )
            save_shap_summary(shap_values, transformed, names, config.figures_dir)
            shap_importance = pd.DataFrame(
                {
                    "feature": names,
                    "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
                }
            ).sort_values("mean_absolute_shap", ascending=False)
            shap_importance.to_csv(
                config.tables_dir / "shap_importance.csv", index=False
            )
            normalized_shap = shap_importance.copy()
            normalized_shap["importance"] = (
                normalized_shap["mean_absolute_shap"]
                / normalized_shap["mean_absolute_shap"].sum()
            )
            report_importance = normalized_shap[["feature", "importance"]]
            shap_status = "generated"
        except (ImportError, ValueError, RuntimeError) as exc:
            shap_status = f"skipped: {exc}"

    if best.estimator is not None:
        joblib.dump(best.estimator, config.outputs_dir / "best_model.joblib")

    metrics_payload = {
        "best_model": best.name,
        "validation_start": str(cutoff.date()),
        "validation_end": str(validation["Date"].max().date()),
        "validation_days": config.validation_days,
        "sample_frac": config.sample_frac,
        "shap_status": shap_status,
        "models": comparison.to_dict(orient="records"),
    }
    _write_json(config.outputs_dir / "metrics.json", metrics_payload)

    print("[7/7] Writing the evidence-grounded model report")
    write_model_report(
        config.report_path,
        comparison,
        segments,
        promotion_lift,
        event_summary,
        report_importance,
        cutoff,
        validation["Date"].max(),
        checks,
    )

    return {
        "best_model": best.name,
        "metrics": best.metrics,
        "report": str(config.report_path),
        "figures": str(config.figures_dir),
        "tables": str(config.tables_dir),
        "shap_status": shap_status,
    }

