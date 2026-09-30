from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rossmann_analytics import ProjectConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Rossmann forecasting and product analytics pipeline."
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=42,
        help="Number of final calendar days reserved for time-based validation.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Recent fraction of each store's history to retain for a faster development run.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip optional SHAP summary generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        project_root=PROJECT_ROOT,
        validation_days=args.validation_days,
        sample_frac=args.sample_frac,
        run_shap=not args.no_shap,
    )
    result = run_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

