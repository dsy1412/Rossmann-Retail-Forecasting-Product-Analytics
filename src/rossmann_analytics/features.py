from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "Sales"
EXCLUDED_MODEL_COLUMNS = {"Date", "Sales", "Customers"}

CATEGORICAL_FEATURES = [
    "Store",
    "StoreType",
    "Assortment",
    "StateHoliday",
    "PromoInterval",
    "competition_band",
]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date = result["Date"]
    result["year"] = date.dt.year.astype("int16")
    result["month"] = date.dt.month.astype("int8")
    result["day"] = date.dt.day.astype("int8")
    result["week_of_year"] = date.dt.isocalendar().week.astype("int16")
    result["day_of_year"] = date.dt.dayofyear.astype("int16")
    result["quarter"] = date.dt.quarter.astype("int8")
    result["is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype("int8")
    result["is_month_start"] = date.dt.is_month_start.astype("int8")
    result["is_month_end"] = date.dt.is_month_end.astype("int8")
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["DayOfWeek"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["DayOfWeek"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["trend_days"] = (date - date.min()).dt.days.astype("int32")
    result["is_state_holiday"] = (result["StateHoliday"] != "none").astype("int8")
    result["is_any_holiday"] = (
        (result["is_state_holiday"] == 1) | (result["SchoolHoliday"] == 1)
    ).astype("int8")
    return result


def add_competition_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    distance = pd.to_numeric(result["CompetitionDistance"], errors="coerce")
    result["competition_distance_missing"] = distance.isna().astype("int8")
    median_distance = float(distance.median()) if distance.notna().any() else 0.0
    result["CompetitionDistance"] = distance.fillna(median_distance).clip(lower=0)
    result["competition_distance_log"] = np.log1p(result["CompetitionDistance"])

    month = pd.to_numeric(result.get("CompetitionOpenSinceMonth"), errors="coerce")
    year = pd.to_numeric(result.get("CompetitionOpenSinceYear"), errors="coerce")
    months_open = (
        (result["Date"].dt.year - year) * 12
        + result["Date"].dt.month
        - month
    )
    result["competition_open_months"] = months_open.where(months_open >= 0, 0).fillna(0)
    result["competition_history_missing"] = (month.isna() | year.isna()).astype("int8")

    result["competition_band"] = pd.cut(
        result["CompetitionDistance"],
        bins=[-np.inf, 500, 2_000, 10_000, np.inf],
        labels=["very_close", "close", "medium", "far"],
    ).astype(str)
    return result


def _promo2_start_date(frame: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(frame.get("Promo2SinceYear"), errors="coerce").astype("Int64")
    week = pd.to_numeric(frame.get("Promo2SinceWeek"), errors="coerce").astype("Int64")
    iso_string = year.astype(str) + "-W" + week.astype(str).str.zfill(2) + "-1"
    valid = year.notna() & week.notna()
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    result.loc[valid] = pd.to_datetime(
        iso_string.loc[valid], format="%G-W%V-%u", errors="coerce"
    )
    return result


def add_promotion_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Promo2"] = pd.to_numeric(result.get("Promo2", 0), errors="coerce").fillna(0)
    promo2_start = _promo2_start_date(result)
    current_month = result["Date"].dt.strftime("%b")
    interval = result.get("PromoInterval", pd.Series("", index=result.index)).fillna("")
    month_is_eligible = [month in months.split(",") for month, months in zip(current_month, interval)]
    result["promo2_active"] = (
        (result["Promo2"] == 1)
        & promo2_start.notna()
        & (result["Date"] >= promo2_start)
        & pd.Series(month_is_eligible, index=result.index)
    ).astype("int8")
    return result


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add target history using only observations before each row."""

    result = frame.sort_values(["Store", "Date"]).copy()
    store_group = result.groupby("Store", sort=False)

    for lag in (1, 7, 14, 28):
        result[f"sales_lag_{lag}"] = store_group[TARGET_COLUMN].shift(lag)

    prior_sales = store_group[TARGET_COLUMN].shift(1)
    for window in (7, 14, 28):
        rolling = prior_sales.groupby(result["Store"], sort=False).rolling(
            window=window,
            min_periods=max(2, window // 3),
        )
        result[f"sales_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        result[f"sales_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

    result["sales_trend_7_vs_28"] = (
        result["sales_roll_mean_7"] / result["sales_roll_mean_28"].replace(0, np.nan) - 1
    )
    result["promo_previous_day"] = store_group["Promo"].shift(1)
    result["promo_previous_week"] = store_group["Promo"].shift(7)
    return result


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_calendar_features(frame)
    result = add_competition_features(result)
    result = add_promotion_features(result)
    result = add_lag_features(result)
    return result.replace([np.inf, -np.inf], np.nan)


def model_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    features = [column for column in frame.columns if column not in EXCLUDED_MODEL_COLUMNS]
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [column for column in features if column not in categorical]
    return features, numeric, categorical

