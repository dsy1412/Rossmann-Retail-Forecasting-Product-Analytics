from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .metrics import regression_metrics


@dataclass
class ModelResult:
    name: str
    estimator: Any
    predictions: np.ndarray
    metrics: dict[str, float]


def time_based_split(
    frame: pd.DataFrame,
    validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if validation_days < 1:
        raise ValueError("validation_days must be positive.")

    max_date = frame["Date"].max()
    cutoff = max_date - pd.Timedelta(days=validation_days - 1)
    train = frame.loc[frame["Date"] < cutoff].copy()
    validation = frame.loc[frame["Date"] >= cutoff].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "The time split produced an empty partition. Reduce validation_days "
            "or provide more history."
        )
    return train, validation, cutoff


def historical_average_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> np.ndarray:
    """Store/day-of-week baseline with safe fallbacks for unseen combinations."""

    open_train = train.loc[(train["Open"] == 1) & (train["Sales"] > 0)]
    by_store_day = open_train.groupby(["Store", "DayOfWeek"])["Sales"].mean()
    by_store = open_train.groupby("Store")["Sales"].mean()
    global_average = float(open_train["Sales"].mean())

    lookup = pd.MultiIndex.from_frame(validation[["Store", "DayOfWeek"]])
    predictions = by_store_day.reindex(lookup).to_numpy(dtype=float)
    store_fallback = validation["Store"].map(by_store).fillna(global_average).to_numpy()
    predictions = np.where(np.isnan(predictions), store_fallback, predictions)
    return np.where(validation["Open"].to_numpy() == 0, 0.0, predictions)


def _linear_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)]
    )


def _tree_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)],
        sparse_threshold=0,
    )


def _tree_estimator(random_state: int) -> tuple[str, Any]:
    try:
        from lightgbm import LGBMRegressor

        return (
            "lightgbm",
            LGBMRegressor(
                objective="regression_l1",
                n_estimators=700,
                learning_rate=0.04,
                num_leaves=48,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.2,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        )
    except ImportError:
        return (
            "random_forest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=24,
                min_samples_leaf=8,
                max_features=0.8,
                random_state=random_state,
                n_jobs=-1,
            ),
        )


def _fit_model(
    name: str,
    pipeline: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> ModelResult:
    training_rows = (train["Open"] == 1) & (train["Sales"] > 0)
    estimator = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    estimator.fit(train.loc[training_rows, features], train.loc[training_rows, "Sales"])
    predictions = np.clip(estimator.predict(validation[features]), 0, None)
    predictions = np.where(validation["Open"].to_numpy() == 0, 0.0, predictions)
    return ModelResult(
        name=name,
        estimator=estimator,
        predictions=predictions,
        metrics=regression_metrics(validation["Sales"], predictions),
    )


def train_and_evaluate_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> list[ModelResult]:
    results: list[ModelResult] = []

    baseline_predictions = historical_average_predictions(train, validation)
    results.append(
        ModelResult(
            name="historical_average",
            estimator=None,
            predictions=baseline_predictions,
            metrics=regression_metrics(validation["Sales"], baseline_predictions),
        )
    )

    linear_pipeline = Pipeline(
        [
            ("preprocess", _linear_preprocessor(numeric_features, categorical_features)),
            ("model", Ridge(alpha=3.0)),
        ]
    )
    results.append(
        _fit_model("ridge", linear_pipeline, train, validation, features)
    )

    tree_name, tree_model = _tree_estimator(random_state)
    tree_pipeline = Pipeline(
        [
            ("preprocess", _tree_preprocessor(numeric_features, categorical_features)),
            ("model", tree_model),
        ]
    )
    results.append(
        _fit_model(tree_name, tree_pipeline, train, validation, features)
    )
    return results


def model_comparison(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("rmspe")


def best_model_result(results: list[ModelResult]) -> ModelResult:
    return min(results, key=lambda result: result.metrics["rmspe"])


def feature_importance_table(result: ModelResult) -> pd.DataFrame:
    if result.estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    fitted_pipeline = result.estimator.regressor_
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    table = pd.DataFrame({"feature": names, "importance": importance})
    total = table["importance"].sum()
    if total > 0:
        table["importance"] = table["importance"] / total
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_values_for_tree_model(
    result: ModelResult,
    validation_features: pd.DataFrame,
    max_rows: int = 2_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return SHAP values in transformed feature space for supported tree models."""

    if result.estimator is None or result.name not in {"lightgbm"}:
        raise ValueError("SHAP is only enabled for the fitted LightGBM model.")

    import shap

    sample = validation_features.sample(
        n=min(max_rows, len(validation_features)),
        random_state=random_state,
    )
    fitted_pipeline = result.estimator.regressor_
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    transformed = np.asarray(preprocess.transform(sample))
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(transformed))
    return values, transformed, list(preprocess.get_feature_names_out())

