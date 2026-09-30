from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ProjectConfig


TRAIN_REQUIRED_COLUMNS = {
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}

STORE_REQUIRED_COLUMNS = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download the Kaggle Rossmann files and place "
            f"them in {path.parent}. See data/README.md."
        )


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_raw_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the competition training and store tables."""

    train_path = config.raw_data_dir / "train.csv"
    store_path = config.raw_data_dir / "store.csv"
    _require_file(train_path)
    _require_file(store_path)

    train = pd.read_csv(train_path, low_memory=False, parse_dates=["Date"])
    store = pd.read_csv(store_path, low_memory=False)
    _require_columns(train, TRAIN_REQUIRED_COLUMNS, "train.csv")
    _require_columns(store, STORE_REQUIRED_COLUMNS, "store.csv")

    if train.empty or store.empty:
        raise ValueError("The Rossmann input files must not be empty.")

    return train, store


def prepare_training_data(
    train: pd.DataFrame,
    store: pd.DataFrame,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """Normalize source values, join store metadata, and optionally sample history."""

    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in the interval (0, 1].")

    frame = train.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["StateHoliday"] = (
        frame["StateHoliday"]
        .astype(str)
        .str.lower()
        .replace(
            {
                "0": "none",
                "0.0": "none",
                "nan": "none",
                "a": "public",
                "b": "easter",
                "c": "christmas",
            }
        )
    )
    frame["Open"] = frame["Open"].fillna(1).astype("int8")
    frame["Promo"] = frame["Promo"].fillna(0).astype("int8")
    frame["SchoolHoliday"] = frame["SchoolHoliday"].fillna(0).astype("int8")

    store_frame = store.copy()
    store_frame["StoreType"] = store_frame["StoreType"].fillna("unknown").astype(str)
    store_frame["Assortment"] = store_frame["Assortment"].fillna("unknown").astype(str)

    frame = frame.merge(store_frame, on="Store", how="left", validate="many_to_one")
    frame = frame.sort_values(["Store", "Date"]).reset_index(drop=True)

    if sample_frac < 1:
        group_sizes = frame.groupby("Store")["Store"].transform("size")
        group_position = frame.groupby("Store").cumcount()
        rows_to_keep = np.maximum(60, np.ceil(group_sizes * sample_frac)).astype(int)
        frame = frame.loc[group_position >= (group_sizes - rows_to_keep)].reset_index(drop=True)

    return frame


def data_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact audit table for missingness and cardinality."""

    return pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing_count": frame.isna().sum().to_numpy(),
            "missing_pct": (frame.isna().mean().mul(100).round(3)).to_numpy(),
            "unique_values": [frame[column].nunique(dropna=True) for column in frame],
        }
    ).sort_values(["missing_pct", "column"], ascending=[False, True])


def data_quality_checks(frame: pd.DataFrame) -> dict[str, int | float | str]:
    duplicate_keys = int(frame.duplicated(["Store", "Date"]).sum())
    negative_sales = int((frame["Sales"] < 0).sum())
    closed_with_sales = int(((frame["Open"] == 0) & (frame["Sales"] > 0)).sum())
    return {
        "rows": int(len(frame)),
        "stores": int(frame["Store"].nunique()),
        "start_date": str(frame["Date"].min().date()),
        "end_date": str(frame["Date"].max().date()),
        "duplicate_store_dates": duplicate_keys,
        "negative_sales_rows": negative_sales,
        "closed_store_positive_sales_rows": closed_with_sales,
    }

