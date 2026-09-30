from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmspe(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared percentage error, excluding zero actual values."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual != 0)
    if not valid.any():
        return float("nan")
    percentage_error = (actual[valid] - predicted[valid]) / actual[valid]
    return float(np.sqrt(np.mean(np.square(percentage_error))))


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.clip(np.asarray(y_pred, dtype=float), a_min=0, a_max=None)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "rmspe": rmspe(actual, predicted),
    }


def segment_performance(
    frame: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    segment_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for segment in segment_columns:
        if segment not in frame:
            continue
        for value, group in frame.groupby(segment, dropna=False, observed=True):
            metrics = regression_metrics(group[actual_column], group[prediction_column])
            records.append(
                {
                    "segment": segment,
                    "value": str(value),
                    "observations": int(len(group)),
                    **metrics,
                }
            )

    return pd.DataFrame.from_records(records).sort_values(
        ["segment", "rmspe"], ascending=[True, False]
    )

