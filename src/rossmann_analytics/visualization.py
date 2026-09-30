from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = ["#2563EB", "#0F766E", "#D97706", "#B91C1C", "#6D28D9"]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
        }
    )


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()


def save_eda_figures(frame: pd.DataFrame, figures_dir: Path) -> None:
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    open_sales = frame.loc[(frame["Open"] == 1) & (frame["Sales"] > 0)].copy()

    upper = open_sales["Sales"].quantile(0.995)
    plt.figure(figsize=(9, 5))
    sns.histplot(open_sales["Sales"].clip(upper=upper), bins=60, color=COLORS[0])
    plt.title("Distribution of Daily Sales on Open Store-Days")
    plt.xlabel("Sales (clipped at 99.5th percentile for display)")
    _save(figures_dir / "sales_distribution.png")

    monthly = open_sales.set_index("Date").resample("MS")["Sales"].mean().reset_index()
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="Date", y="Sales", color=COLORS[1], linewidth=2)
    plt.title("Average Daily Sales Over Time")
    _save(figures_dir / "monthly_sales_trend.png")

    for column, filename, title in (
        ("StoreType", "sales_by_store_type.png", "Sales by Store Type"),
        ("Assortment", "sales_by_assortment.png", "Sales by Assortment"),
        ("Promo", "sales_by_promotion.png", "Sales by Promotion Status"),
        ("DayOfWeek", "sales_by_day_of_week.png", "Sales by Day of Week"),
        ("StateHoliday", "sales_by_state_holiday.png", "Sales Around State Holidays"),
        ("SchoolHoliday", "sales_by_school_holiday.png", "Sales by School Holiday Status"),
    ):
        summary = (
            open_sales.groupby(column, observed=True)["Sales"]
            .agg(mean_sales="mean", median_sales="median", store_days="size")
            .reset_index()
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x=column, y="mean_sales", color=COLORS[0])
        plt.title(title)
        plt.ylabel("Mean daily sales")
        _save(figures_dir / filename)

    competition = open_sales.loc[open_sales["CompetitionDistance"].notna()].copy()
    competition["distance_decile"] = pd.qcut(
        competition["CompetitionDistance"], q=10, duplicates="drop"
    )
    distance_summary = (
        competition.groupby("distance_decile", observed=True)
        .agg(competition_distance=("CompetitionDistance", "median"), sales=("Sales", "mean"))
        .reset_index(drop=True)
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=distance_summary,
        x="competition_distance",
        y="sales",
        marker="o",
        color=COLORS[2],
    )
    plt.xscale("log")
    plt.title("Mean Sales by Competition-Distance Decile")
    plt.xlabel("Median competition distance (log scale)")
    plt.ylabel("Mean daily sales")
    _save(figures_dir / "competition_distance_vs_sales.png")

    missing = frame.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing.loc[missing > 0].head(15)
    plt.figure(figsize=(9, max(4, len(missing) * 0.35)))
    if missing.empty:
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        plt.axis("off")
    else:
        sns.barplot(x=missing.values, y=missing.index, color=COLORS[3])
        plt.xlabel("Missing values (%)")
        plt.ylabel("")
    plt.title("Input Data Missingness")
    _save(figures_dir / "missing_values.png")


def save_model_comparison_figure(comparison: pd.DataFrame, figures_dir: Path) -> None:
    _style()
    plot_data = comparison.sort_values("rmspe")
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=plot_data, x="rmspe", y="model", color=COLORS[0])
    plt.title("Time-Holdout Model Comparison")
    plt.xlabel("RMSPE (lower is better)")
    plt.ylabel("")
    _save(figures_dir / "model_comparison_rmspe.png")


def save_feature_importance_figure(importance: pd.DataFrame, figures_dir: Path) -> None:
    if importance.empty:
        return
    _style()
    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top, x="importance", y="feature", color=COLORS[1])
    plt.title("Top Predictive Features")
    plt.xlabel("Normalized importance")
    plt.ylabel("")
    _save(figures_dir / "feature_importance.png")


def save_shap_summary(
    shap_values: np.ndarray,
    transformed_features: np.ndarray,
    feature_names: list[str],
    figures_dir: Path,
) -> None:
    import shap

    plt.figure()
    shap.summary_plot(
        shap_values,
        transformed_features,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary for LightGBM Predictions")
    _save(figures_dir / "shap_summary.png")

