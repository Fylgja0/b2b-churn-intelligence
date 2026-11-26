"""
Unified Baseline Modeling & Experiment Runner.

Executes scenario-configured churn modeling experiments (`region_included` or
`region_excluded`) through a single shared pipeline without duplicating model
or preprocessing code.

Outputs per scenario:
  - Reports (`ml/reports/modeling/experiments/<scenario>/`):
      * `experiment_summary.json`
      * `metrics_summary.csv`
      * `experiment_report.md`
      * `oof_<target>_<model>.csv`
      * `figures/pr_curve_<target>.png`, `figures/roc_curve_<target>.png`
  - Trained model artifacts (`ml/artifacts/models/<scenario>/`):
      * `<target>_<model>.joblib` (full-cohort fitted pipelines for downstream use)

Note: This module provides the infrastructure and CLI entrypoint. Real baseline
training runs are intentionally triggered explicitly by the user in the next phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone

from ml.src.evaluation.compare_experiments import compare_scenarios
from ml.src.evaluation.metrics import (
    METRIC_PRIORITY,
    PRIMARY_METRIC,
    PRIMARY_METRIC_DISPLAY_NAME,
)
from ml.src.evaluation.plots import plot_pr_curves, plot_roc_curves
from ml.src.models.cross_validation import (
    CVResult,
    generate_cv_splits,
    run_cross_validation,
)
from ml.src.models.experiment_config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_BASELINE_MODELS,
    DEFAULT_DATASET_PATH,
    DEFAULT_EXPERIMENTS_REPORTS_DIR,
    DEFAULT_MANIFEST_PATH,
    ExperimentConfig,
    VALID_SCENARIOS,
    build_experiment_config,
    to_relative_path,
)
from ml.src.models.model_registry import ModelRegistry, get_default_registry
from ml.src.models.preprocessing import prepare_feature_matrix

logger = logging.getLogger(__name__)


def compute_dataset_fingerprint(dataset_path: Path) -> str:
    """
    Compute a deterministic SHA-256 hex digest of the modeling CSV file bytes.

    Guarantees identical file content and exact row ordering when comparing
    experiments across scenarios.
    """
    hasher = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_metrics_summary_table(cv_results: list[CVResult]) -> pd.DataFrame:
    """Build a flat metrics summary table across all targets and models."""
    rows: list[dict[str, Any]] = []
    for res in cv_results:
        row: dict[str, Any] = {
            "scenario": res.scenario,
            "target": res.target,
            "model": res.model_name,
            "n_samples": res.class_distribution["n_samples"],
            "n_positive": res.class_distribution["n_positive"],
            "positive_rate_pct": res.class_distribution["positive_rate_pct"],
            "decision_threshold": res.decision_threshold,
            "threshold_optimized": res.threshold_optimized,
        }
        for metric in METRIC_PRIORITY:
            row[f"{metric}_mean"] = res.summary_metrics[metric]["mean"]
            row[f"{metric}_std"] = res.summary_metrics[metric]["std"]
            row[f"{metric}_oof"] = res.oof_metrics[metric]
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["target", f"{PRIMARY_METRIC}_mean"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return df


def _render_experiment_markdown(
    config: ExperimentConfig,
    dataset_row_count: int,
    dataset_fingerprint: str,
    cv_results: list[CVResult],
    metrics_df: pd.DataFrame,
) -> str:
    """Render a scenario experiment report in Markdown."""
    selected, categorical, numeric = config.get_scenario_features()
    lines: list[str] = [
        f"# Baseline Experiment Report — `{config.scenario}`",
        "",
        "## Experiment Configuration",
        f"- **Scenario**: `{config.scenario}`",
        f"- **Observation Date (T0)**: `{config.feature_schema.observation_date}`",
        f"- **Dataset Cutoff Date**: `{config.feature_schema.dataset_cutoff_date}`",
        f"- **Dataset Rows**: {dataset_row_count}",
        f"- **Dataset Fingerprint (SHA-256)**: `{dataset_fingerprint}`",
        f"- **Primary Target**: `{config.primary_target}`",
        f"- **Targets Evaluated**: " + ", ".join(f"`{t}`" for t in config.targets),
        f"- **Selected Features ({len(selected)})**: " + ", ".join(f"`{c}`" for c in selected),
        f"- **Categorical Features ({len(categorical)})**: " + ", ".join(f"`{c}`" for c in categorical),
        f"- **Numeric Features ({len(numeric)})**: " + ", ".join(f"`{c}`" for c in numeric),
        f"- **Dropped Scenario Features**: " + ", ".join(f"`{c}`" for c in config.get_dropped_features()),
        f"- **Cross-Validation**: `StratifiedKFold(n_splits={config.n_splits}, shuffle={config.shuffle}, random_state={config.random_state})`",
        f"- **Primary Metric**: `{PRIMARY_METRIC}` — {PRIMARY_METRIC_DISPLAY_NAME}",
        f"- **Decision Threshold**: `{config.decision_threshold}` (**not optimized**; used for Recall, Precision, and F1)",
        "- **Model Selection Note**: Accuracy is not used for model selection due to class imbalance.",
        "",
    ]

    # Fold class distribution per target (from first model per target)
    seen_targets: set[str] = set()
    for res in cv_results:
        if res.target in seen_targets:
            continue
        seen_targets.add(res.target)
        dist = res.class_distribution
        lines.extend(
            [
                f"## Target: `{res.target}`",
                f"- Total samples: **{dist['n_samples']}** | Positives: **{dist['n_positive']}** ({dist['positive_rate_pct']:.2f}%) | Negatives: **{dist['n_negative']}**",
                "",
                "### Fold Class Distribution",
                "| Fold | Train Samples | Train Positives | Val Samples | Val Positives | Val Positive Rate (%) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fd in res.fold_distributions:
            lines.append(
                f"| {fd['fold']} | {fd['train_n_samples']} | {fd['train_n_positive']} | "
                f"{fd['val_n_samples']} | {fd['val_n_positive']} | {fd['val_positive_rate_pct']:.2f}% |"
            )

        lines.extend(
            [
                "",
                "### Cross-Validation Metrics (Mean ± Std)",
                "| Model | Average Precision (AP) | ROC-AUC | Recall (@0.5) | Precision (@0.5) | F1 (@0.5) | Brier Score |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        target_df = metrics_df[metrics_df["target"] == res.target]
        for _, r in target_df.iterrows():
            lines.append(
                f"| `{r['model']}` | "
                f"{r['average_precision_mean']:.4f} ± {r['average_precision_std']:.4f} | "
                f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
                f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f} | "
                f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f} | "
                f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | "
                f"{r['brier_score_mean']:.4f} ± {r['brier_score_std']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_experiment(
    config: ExperimentConfig,
    registry: ModelRegistry | None = None,
    save_artifacts: bool = True,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """
    Execute cross-validation experiments for a single scenario configuration.

    Parameters
    ----------
    config:
        Validated `ExperimentConfig` (for either `region_included` or `region_excluded`).
    registry:
        Optional custom `ModelRegistry`; defaults to `get_default_registry()`.
    save_artifacts:
        If True, fit full-dataset pipelines and persist them via `joblib` under
        `config.artifacts_dir`.
    generate_plots:
        If True, generate OOF PR and ROC curve PNGs under `config.reports_dir/figures`.
    """
    active_registry = registry or get_default_registry()
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Modeling dataset not found at: {dataset_path}")

    dataset_fingerprint = compute_dataset_fingerprint(dataset_path)
    df = pd.read_csv(dataset_path)
    config.validate_dataframe(df)

    schema = config.feature_schema
    selected_features, categorical_features, numeric_features = config.get_scenario_features()

    X = prepare_feature_matrix(
        df=df,
        feature_columns=selected_features,
        categorical_columns=categorical_features,
        metadata_columns=schema.metadata_columns,
        target_columns=schema.target_columns,
        excluded_columns=schema.excluded_columns,
    )
    metadata_df = df[list(schema.metadata_columns)].copy()

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    if save_artifacts:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    cv_results: list[CVResult] = []
    oof_files: dict[str, str] = {}
    artifact_files: dict[str, str] = {}
    plot_files: list[str] = []

    for target in config.targets:
        y = df[target]
        splits = generate_cv_splits(
            y=y,
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=config.random_state,
        )
        oof_by_model: dict[str, pd.DataFrame] = {}

        for model_name in config.models:
            logger.info(
                "Running CV | scenario=%s | target=%s | model=%s",
                config.scenario,
                target,
                model_name,
            )
            pipeline = active_registry.create_pipeline(
                name=model_name,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                random_state=config.random_state,
            )

            cv_result = run_cross_validation(
                pipeline=pipeline,
                X=X,
                y=y,
                metadata_df=metadata_df,
                scenario=config.scenario,
                target=target,
                model_name=model_name,
                splits=splits,
                n_splits=config.n_splits,
                shuffle=config.shuffle,
                random_state=config.random_state,
                decision_threshold=config.decision_threshold,
            )
            cv_results.append(cv_result)
            oof_by_model[model_name] = cv_result.oof_predictions

            oof_path = config.reports_dir / f"oof_{target}_{model_name}.csv"
            cv_result.oof_predictions.to_csv(oof_path, index=False)
            oof_files[f"{target}__{model_name}"] = to_relative_path(oof_path)

            if save_artifacts:
                fitted_full = clone(pipeline)
                fitted_full.fit(X, y.to_numpy(dtype=int))
                artifact_path = config.artifacts_dir / f"{target}_{model_name}.joblib"
                joblib.dump(fitted_full, artifact_path)
                artifact_files[f"{target}__{model_name}"] = to_relative_path(artifact_path)

        if generate_plots:
            figures_dir = config.reports_dir / "figures"
            pr_path = plot_pr_curves(
                oof_by_model=oof_by_model,
                target=target,
                output_path=figures_dir / f"pr_curve_{target}.png",
                scenario=config.scenario,
            )
            roc_path = plot_roc_curves(
                oof_by_model=oof_by_model,
                target=target,
                output_path=figures_dir / f"roc_curve_{target}.png",
                scenario=config.scenario,
            )
            plot_files.extend([to_relative_path(pr_path), to_relative_path(roc_path)])

    metrics_df = _build_metrics_summary_table(cv_results)
    metrics_csv_path = config.reports_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    report_md = _render_experiment_markdown(
        config=config,
        dataset_row_count=len(df),
        dataset_fingerprint=dataset_fingerprint,
        cv_results=cv_results,
        metrics_df=metrics_df,
    )
    report_md_path = config.reports_dir / "experiment_report.md"
    report_md_path.write_text(report_md, encoding="utf-8")

    summary_payload: dict[str, Any] = {
        "config": config.to_metadata_dict(),
        "dataset_row_count": int(len(df)),
        "dataset_fingerprint": dataset_fingerprint,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_display_name": PRIMARY_METRIC_DISPLAY_NAME,
        "metric_priority": list(METRIC_PRIORITY),
        "threshold_note": (
            f"Precision, Recall, and F1 are evaluated at decision_threshold={config.decision_threshold} "
            "(unoptimized baseline threshold)."
        ),
        "results": [r.to_summary_dict() for r in cv_results],
        "oof_files": oof_files,
        "artifact_files": artifact_files,
        "plot_files": plot_files,
        "metrics_csv_path": to_relative_path(metrics_csv_path),
        "report_md_path": to_relative_path(report_md_path),
    }

    summary_json_path = config.reports_dir / "experiment_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info(
        "Completed scenario '%s'. Summary saved to %s",
        config.scenario,
        to_relative_path(summary_json_path),
    )
    return summary_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline B2B churn modeling experiments."
    )
    parser.add_argument(
        "--scenario",
        choices=[*VALID_SCENARIOS, "all"],
        required=True,
        help="Experiment scenario to run (`region_included`, `region_excluded`, or `all`).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional subset of target columns to evaluate (defaults to all 3 manifest targets).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_BASELINE_MODELS),
        help="Registered model names to evaluate.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to churn_modeling_dataset.csv.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to feature_manifest.json.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Base directory for saved model artifacts.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_REPORTS_DIR,
        help="Base directory for experiment reports.",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not serialize fitted full-cohort model binaries.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PR/ROC curve charts.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Automatically run scenario comparison after running `--scenario all`.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    scenarios = list(VALID_SCENARIOS) if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        config = build_experiment_config(
            scenario=scenario,
            targets=args.targets,
            models=args.models,
            dataset_path=args.dataset_path,
            manifest_path=args.manifest_path,
            artifacts_base_dir=args.artifacts_dir,
            reports_base_dir=args.reports_dir,
        )
        run_experiment(
            config=config,
            save_artifacts=not args.no_save_artifacts,
            generate_plots=not args.no_plots,
        )

    if args.compare and args.scenario == "all":
        compare_scenarios(
            reports_base_dir=args.reports_dir,
            generate_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
