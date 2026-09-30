from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    """Paths and reproducibility settings for a pipeline run."""

    project_root: Path = PROJECT_ROOT
    validation_days: int = 42
    random_state: int = 42
    sample_frac: float = 1.0
    run_shap: bool = True

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "reports" / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.project_root / "reports" / "tables"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def report_path(self) -> Path:
        return self.project_root / "reports" / "model_report.md"

    def ensure_directories(self) -> None:
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.figures_dir,
            self.tables_dir,
            self.outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

