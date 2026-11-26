"""
Experiment Configuration & Manifest-Driven Feature Schema Management.

Reads `ml/reports/modeling/feature_manifest.json` as the single source of truth
for metadata, feature, target, and excluded (leakage/unverified) columns.
Defines the two comparative modeling scenarios:
  - `region_included`: keeps `region` (missing values imputed as "Unknown" inside
    the pipeline) and drops `region_missing_flag` to avoid duplicate missingness signal.
  - `region_excluded`: drops both `region` and `region_missing_flag` while keeping
    all observations, targets, CV splits, algorithms, and metrics identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
REPO_ROOT = ML_DIR.parent

DEFAULT_DATASET_PATH = ML_DIR / "data" / "processed" / "modeling" / "churn_modeling_dataset.csv"
DEFAULT_MANIFEST_PATH = ML_DIR / "reports" / "modeling" / "feature_manifest.json"
DEFAULT_ARTIFACTS_DIR = ML_DIR / "artifacts" / "models"
DEFAULT_EXPERIMENTS_REPORTS_DIR = ML_DIR / "reports" / "modeling" / "experiments"

SCENARIO_REGION_INCLUDED = "region_included"
SCENARIO_REGION_EXCLUDED = "region_excluded"
VALID_SCENARIOS: tuple[str, ...] = (SCENARIO_REGION_INCLUDED, SCENARIO_REGION_EXCLUDED)

# Features dropped per scenario relative to feature_manifest.json:
# - In `region_included`, missing `region` values are encoded as "Unknown" inside
#   the categorical pipeline, so `region_missing_flag` is omitted to prevent
#   duplicating the missingness indicator.
# - In `region_excluded`, both `region` and `region_missing_flag` are omitted.
SCENARIO_DROPPED_FEATURES: dict[str, tuple[str, ...]] = {
    SCENARIO_REGION_INCLUDED: ("region_missing_flag",),
    SCENARIO_REGION_EXCLUDED: ("region", "region_missing_flag"),
}

PRIMARY_TARGET = "target_churn_365d"
DECISION_THRESHOLD = 0.5
RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 5

DEFAULT_BASELINE_MODELS: tuple[str, ...] = (
    "dummy",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
)


def to_relative_path(path: Path | str, base_dir: Path | None = None) -> str:
    """Convert an absolute path to a portable repository-relative POSIX path."""
    p = Path(path)
    base = Path(base_dir or REPO_ROOT).resolve()
    try:
        return p.resolve().relative_to(base).as_posix()
    except (ValueError, RuntimeError):
        return p.as_posix()


@dataclass(frozen=True)
class FeatureSchema:
    """Schema parsed directly from `feature_manifest.json`."""

    version: str
    observation_date: str
    dataset_cutoff_date: str
    metadata_columns: tuple[str, ...]
    all_feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "observation_date": self.observation_date,
            "dataset_cutoff_date": self.dataset_cutoff_date,
            "metadata_columns": list(self.metadata_columns),
            "all_feature_columns": list(self.all_feature_columns),
            "categorical_columns": list(self.categorical_columns),
            "numeric_columns": list(self.numeric_columns),
            "target_columns": list(self.target_columns),
            "excluded_columns": list(self.excluded_columns),
        }


def load_feature_schema(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> FeatureSchema:
    """Load and validate the feature schema from `feature_manifest.json`."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Feature manifest not found at: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    metadata_columns = tuple(col["name"] for col in manifest["metadata_columns"])
    feature_entries = manifest["feature_columns"]
    all_feature_columns = tuple(col["name"] for col in feature_entries)
    categorical_columns = tuple(
        col["name"] for col in feature_entries if col["data_type"] == "string"
    )
    numeric_columns = tuple(
        col["name"] for col in feature_entries if col["data_type"] in ("integer", "float")
    )
    target_columns = tuple(col["name"] for col in manifest["target_columns"])
    excluded_columns = tuple(col["name"] for col in manifest.get("excluded_columns", []))

    # Validate partition completeness and disjointness
    if set(categorical_columns) | set(numeric_columns) != set(all_feature_columns):
        unknown_types = set(all_feature_columns) - (set(categorical_columns) | set(numeric_columns))
        raise ValueError(f"Feature manifest contains unsupported data_type for: {sorted(unknown_types)}")

    if set(metadata_columns) & set(all_feature_columns):
        raise ValueError("Manifest metadata_columns and feature_columns must be disjoint.")
    if set(target_columns) & set(all_feature_columns):
        raise ValueError("Manifest target_columns and feature_columns must be disjoint.")
    if set(metadata_columns) & set(target_columns):
        raise ValueError("Manifest metadata_columns and target_columns must be disjoint.")

    return FeatureSchema(
        version=str(manifest.get("version", "unknown")),
        observation_date=str(manifest["observation_date"]),
        dataset_cutoff_date=str(manifest["dataset_cutoff_date"]),
        metadata_columns=metadata_columns,
        all_feature_columns=all_feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target_columns=target_columns,
        excluded_columns=excluded_columns,
    )


def resolve_scenario_features(
    scenario: str,
    schema: FeatureSchema,
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve `(selected_features, categorical_features, numeric_features)` for a scenario
    using `FeatureSchema` as the authoritative source.
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unsupported scenario '{scenario}'. Valid scenarios: {VALID_SCENARIOS}")

    dropped = set(SCENARIO_DROPPED_FEATURES[scenario])
    selected = [col for col in schema.all_feature_columns if col not in dropped]
    categorical = [col for col in schema.categorical_columns if col not in dropped]
    numeric = [col for col in schema.numeric_columns if col not in dropped]

    return selected, categorical, numeric


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for a modeling experiment scenario."""

    scenario: str
    targets: tuple[str, ...]
    models: tuple[str, ...] = DEFAULT_BASELINE_MODELS
    primary_target: str = PRIMARY_TARGET
    n_splits: int = DEFAULT_CV_SPLITS
    shuffle: bool = True
    random_state: int = RANDOM_STATE
    decision_threshold: float = DECISION_THRESHOLD
    threshold_optimized: bool = False
    dataset_path: Path = DEFAULT_DATASET_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    artifacts_dir: Path = field(default_factory=lambda: DEFAULT_ARTIFACTS_DIR)
    reports_dir: Path = field(default_factory=lambda: DEFAULT_EXPERIMENTS_REPORTS_DIR)
    schema: FeatureSchema | None = None

    def __post_init__(self) -> None:
        if self.scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Invalid scenario '{self.scenario}'. Expected one of {VALID_SCENARIOS}."
            )
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}.")
        if not (0.0 < self.decision_threshold < 1.0):
            raise ValueError(
                f"decision_threshold must be strictly between 0 and 1, got {self.decision_threshold}."
            )
        resolved_schema = self.schema or load_feature_schema(self.manifest_path)
        object.__setattr__(self, "schema", resolved_schema)

        invalid_targets = [t for t in self.targets if t not in resolved_schema.target_columns]
        if invalid_targets:
            raise ValueError(
                f"Invalid target(s) {invalid_targets}. Valid manifest targets: "
                f"{list(resolved_schema.target_columns)}"
            )
        if not self.targets:
            raise ValueError("At least one target column must be specified.")
        if not self.models:
            raise ValueError("At least one model name must be specified.")

    @property
    def feature_schema(self) -> FeatureSchema:
        assert self.schema is not None
        return self.schema

    def get_scenario_features(self) -> tuple[list[str], list[str], list[str]]:
        """Return `(selected_features, categorical_features, numeric_features)`."""
        return resolve_scenario_features(self.scenario, self.feature_schema)

    def get_dropped_features(self) -> list[str]:
        """Return manifest features excluded by this scenario."""
        return list(SCENARIO_DROPPED_FEATURES[self.scenario])

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that a modeling DataFrame contains all required metadata, feature,
        and target columns, and that selected features do not overlap with metadata,
        targets, or manifest-excluded leakage columns.
        """
        schema = self.feature_schema
        selected_features, _, _ = self.get_scenario_features()

        required_columns = (
            set(schema.metadata_columns) | set(selected_features) | set(self.targets)
        )
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        forbidden_in_features = (
            set(schema.metadata_columns)
            | set(schema.target_columns)
            | set(schema.excluded_columns)
        )
        leaked = sorted(set(selected_features) & forbidden_in_features)
        if leaked:
            raise ValueError(
                f"Selected features contain forbidden metadata/target/leakage columns: {leaked}"
            )

        if df.empty:
            raise ValueError("Modeling dataset is empty.")

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize configuration to a portable JSON-compatible dictionary."""
        selected, categorical, numeric = self.get_scenario_features()
        return {
            "scenario": self.scenario,
            "observation_date": self.feature_schema.observation_date,
            "dataset_cutoff_date": self.feature_schema.dataset_cutoff_date,
            "primary_target": self.primary_target,
            "targets": list(self.targets),
            "models": list(self.models),
            "selected_features": selected,
            "categorical_features": categorical,
            "numeric_features": numeric,
            "dropped_features": self.get_dropped_features(),
            "metadata_columns": list(self.feature_schema.metadata_columns),
            "cv_strategy": "StratifiedKFold",
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "threshold_optimized": self.threshold_optimized,
            "dataset_path": to_relative_path(self.dataset_path),
            "manifest_path": to_relative_path(self.manifest_path),
            "artifacts_dir": to_relative_path(self.artifacts_dir),
            "reports_dir": to_relative_path(self.reports_dir),
        }


def build_experiment_config(
    scenario: str,
    targets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    n_splits: int = DEFAULT_CV_SPLITS,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
    decision_threshold: float = DECISION_THRESHOLD,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    artifacts_base_dir: Path = DEFAULT_ARTIFACTS_DIR,
    reports_base_dir: Path = DEFAULT_EXPERIMENTS_REPORTS_DIR,
) -> ExperimentConfig:
    """Factory that builds a validated scenario-specific `ExperimentConfig`."""
    schema = load_feature_schema(manifest_path)
    resolved_targets = tuple(targets) if targets is not None else schema.target_columns
    resolved_models = tuple(models) if models is not None else DEFAULT_BASELINE_MODELS

    return ExperimentConfig(
        scenario=scenario,
        targets=resolved_targets,
        models=resolved_models,
        primary_target=PRIMARY_TARGET,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        decision_threshold=decision_threshold,
        threshold_optimized=False,
        dataset_path=Path(dataset_path),
        manifest_path=Path(manifest_path),
        artifacts_dir=Path(artifacts_base_dir) / scenario,
        reports_dir=Path(reports_base_dir) / scenario,
        schema=schema,
    )
