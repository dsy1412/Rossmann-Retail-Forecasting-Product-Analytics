from __future__ import annotations

import numpy as np
import pandas as pd


def promotion_lift_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate descriptive within-store promotion lift; this is not causal."""

    open_days = frame.loc[(frame["Open"] == 1) & (frame["Sales"] > 0)].copy()
    nonpromo_baseline = (
        open_days.loc[open_days["Promo"] == 0]
        .groupby("Store")["Sales"]
        .mean()
        .rename("store_nonpromo_mean")
    )
    promo_days = open_days.loc[open_days["Promo"] == 1].join(
        nonpromo_baseline, on="Store", how="inner"
    )
    promo_days = promo_days.loc[promo_days["store_nonpromo_mean"] > 0].copy()
    promo_days["observational_lift"] = (
        promo_days["Sales"] / promo_days["store_nonpromo_mean"] - 1
    )

    records: list[dict[str, object]] = []
    segment_definitions = {
        "overall": None,
        "store_type": "StoreType",
        "assortment": "Assortment",
        "competition_band": "competition_band",
    }
    for segment_name, column in segment_definitions.items():
        groups = [("all", promo_days)] if column is None else promo_days.groupby(column, observed=True)
        for value, group in groups:
            records.append(
                {
                    "segment": segment_name,
                    "value": str(value),
                    "promo_store_days": int(len(group)),
                    "stores": int(group["Store"].nunique()),
                    "mean_observational_lift": float(group["observational_lift"].mean()),
                    "median_observational_lift": float(group["observational_lift"].median()),
                    "mean_promo_sales": float(group["Sales"].mean()),
                    "mean_store_nonpromo_baseline": float(group["store_nonpromo_mean"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def root_cause_events(frame: pd.DataFrame, max_events: int = 200) -> pd.DataFrame:
    """Flag extreme store-days relative to each store's trailing 28-day mean."""

    candidates = frame.loc[
        (frame["Open"] == 1)
        & frame["sales_roll_mean_28"].notna()
        & (frame["sales_roll_mean_28"] > 0)
    ].copy()
    candidates["sales_deviation_pct"] = (
        candidates["Sales"] / candidates["sales_roll_mean_28"] - 1
    )
    lower = candidates["sales_deviation_pct"].quantile(0.01)
    upper = candidates["sales_deviation_pct"].quantile(0.99)
    candidates["event_type"] = np.select(
        [candidates["sales_deviation_pct"] <= lower, candidates["sales_deviation_pct"] >= upper],
        ["drop", "spike"],
        default="normal",
    )
    events = candidates.loc[candidates["event_type"] != "normal"].copy()
    events["absolute_deviation"] = events["sales_deviation_pct"].abs()

    columns = [
        "Date",
        "Store",
        "event_type",
        "Sales",
        "sales_roll_mean_28",
        "sales_deviation_pct",
        "Promo",
        "promo2_active",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "competition_band",
    ]
    return (
        events.nlargest(max_events, "absolute_deviation")[columns]
        .sort_values(["event_type", "sales_deviation_pct"])
        .reset_index(drop=True)
    )


def root_cause_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    return (
        events.groupby("event_type")
        .agg(
            events=("Store", "size"),
            stores=("Store", "nunique"),
            median_deviation=("sales_deviation_pct", "median"),
            promo_share=("Promo", "mean"),
            school_holiday_share=("SchoolHoliday", "mean"),
            state_holiday_share=("StateHoliday", lambda values: (values != "none").mean()),
        )
        .reset_index()
    )

