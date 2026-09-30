from __future__ import annotations

import numpy as np
import pandas as pd

from rossmann_analytics.analytics import promotion_lift_analysis, root_cause_events
from rossmann_analytics.data import prepare_training_data
from rossmann_analytics.features import add_lag_features, build_features, model_feature_columns
from rossmann_analytics.metrics import regression_metrics, rmspe
from rossmann_analytics.modeling import (
    best_model_result,
    feature_importance_table,
    time_based_split,
    train_and_evaluate_models,
)


def test_rmspe_excludes_zero_actuals() -> None:
    actual = np.array([100.0, 0.0, 200.0])
    predicted = np.array([90.0, 500.0, 220.0])
    assert np.isclose(rmspe(actual, predicted), 0.1)


def test_regression_metrics_are_nonnegative() -> None:
    metrics = regression_metrics([100, 200], [-10, 210])
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["rmspe"] >= 0


def test_lag_features_do_not_use_current_target() -> None:
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    frame = pd.DataFrame(
        {
            "Store": 1,
            "Date": dates,
            "Sales": np.arange(1, 36, dtype=float),
            "Promo": 0,
        }
    )
    original = add_lag_features(frame)
    modified = frame.copy()
    modified.loc[20, "Sales"] = 999_999
    changed = add_lag_features(modified)

    assert original.loc[20, "sales_lag_1"] == changed.loc[20, "sales_lag_1"]
    assert original.loc[20, "sales_roll_mean_7"] == changed.loc[20, "sales_roll_mean_7"]
    assert original.loc[21, "sales_lag_1"] != changed.loc[21, "sales_lag_1"]


def test_time_split_uses_future_dates_for_validation() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "Sales": 1,
        }
    )
    train, validation, cutoff = time_based_split(frame, validation_days=14)
    assert train["Date"].max() < cutoff
    assert validation["Date"].min() == cutoff
    assert validation["Date"].max() > train["Date"].max()


def test_promotion_lift_uses_store_specific_nonpromo_baseline() -> None:
    frame = pd.DataFrame(
        {
            "Store": [1, 1, 1, 2, 2, 2],
            "Open": 1,
            "Sales": [100, 100, 120, 200, 200, 220],
            "Promo": [0, 0, 1, 0, 0, 1],
            "StoreType": ["a", "a", "a", "b", "b", "b"],
            "Assortment": ["a", "a", "a", "b", "b", "b"],
            "competition_band": ["close"] * 3 + ["far"] * 3,
        }
    )
    result = promotion_lift_analysis(frame)
    overall = result.loc[result["segment"] == "overall"].iloc[0]
    assert np.isclose(overall["mean_observational_lift"], 0.15)


def test_end_to_end_components_on_synthetic_data() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    for store in (1, 2, 3):
        for date in dates:
            day_of_week = date.dayofweek + 1
            promo = int(day_of_week in (1, 4))
            sales = (
                4_000
                + store * 300
                + promo * 600
                + 250 * np.sin(2 * np.pi * day_of_week / 7)
                + rng.normal(0, 80)
            )
            rows.append(
                {
                    "Store": store,
                    "DayOfWeek": day_of_week,
                    "Date": date,
                    "Sales": max(0, sales),
                    "Customers": 500,
                    "Open": 1,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": int(date.month in (7, 8)),
                }
            )
    train = pd.DataFrame(rows)
    store = pd.DataFrame(
        {
            "Store": [1, 2, 3],
            "StoreType": ["a", "b", "c"],
            "Assortment": ["a", "a", "c"],
            "CompetitionDistance": [500, 2_500, np.nan],
            "CompetitionOpenSinceMonth": [1, 6, np.nan],
            "CompetitionOpenSinceYear": [2020, 2021, np.nan],
            "Promo2": [1, 0, 1],
            "Promo2SinceWeek": [1, np.nan, 10],
            "Promo2SinceYear": [2023, np.nan, 2023],
            "PromoInterval": ["Jan,Apr,Jul,Oct", np.nan, "Mar,Jun,Sept,Dec"],
        }
    )

    merged = prepare_training_data(train, store)
    featured = build_features(merged)
    features, numeric, categorical = model_feature_columns(featured)
    training, validation, _ = time_based_split(featured, validation_days=21)
    results = train_and_evaluate_models(
        training,
        validation,
        features,
        numeric,
        categorical,
        random_state=42,
    )

    assert len(results) == 3
    assert all(np.isfinite(result.metrics["rmspe"]) for result in results)
    assert len(best_model_result(results).predictions) == len(validation)
    assert not feature_importance_table(results[-1]).empty
    assert not promotion_lift_analysis(featured).empty
    assert set(root_cause_events(featured)["event_type"].unique()) <= {"drop", "spike"}

