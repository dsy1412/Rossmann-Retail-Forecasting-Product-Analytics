from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rossmann_analytics import ProjectConfig, run_pipeline


def build_synthetic_raw_data(project_root: Path) -> None:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")

    rows = []
    for store in (1, 2, 3):
        for date in dates:
            day_of_week = date.dayofweek + 1
            promo = int(day_of_week in (1, 4))
            open_flag = int(day_of_week != 7)
            sales = open_flag * (
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
                    "Customers": open_flag * 500,
                    "Open": open_flag,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": int(date.month in (7, 8)),
                }
            )
    pd.DataFrame(rows).to_csv(raw_dir / "train.csv", index=False)

    pd.DataFrame(
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
    ).to_csv(raw_dir / "store.csv", index=False)


def run_smoke_test(project_root: Path) -> dict[str, object]:
    build_synthetic_raw_data(project_root)
    result = run_pipeline(
        ProjectConfig(
            project_root=project_root,
            validation_days=21,
            random_state=42,
            run_shap=False,
        )
    )
    required_outputs = [
        project_root / "reports" / "model_report.md",
        project_root / "reports" / "figures" / "sales_distribution.png",
        project_root / "reports" / "figures" / "model_comparison_rmspe.png",
        project_root / "reports" / "tables" / "segment_performance.csv",
        project_root / "reports" / "tables" / "promotion_lift.csv",
        project_root / "outputs" / "metrics.json",
        project_root / "outputs" / "validation_predictions.csv",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    assert not missing, f"Missing smoke-test outputs: {missing}"
    return result


def test_complete_pipeline(tmp_path: Path) -> None:
    run_smoke_test(tmp_path)

