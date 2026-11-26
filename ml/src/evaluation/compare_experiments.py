"""
Scenario Comparison Module (`region_included` vs `region_excluded`).

Loads completed experiment summaries from both scenarios, strictly verifies that
comparison invariants hold (expected scenario names, identical SHA-256 dataset
fingerprint, row count, CV settings `n_splits`/`shuffle`/`random_state`,
`decision_threshold`, identical model and target lists, complete `(target, model)`
pair coverage, and identical target class distributions), computes per-metric
deltas (`region_included - region_excluded`), ranks models by Average Precision
(AP), and writes comparison reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.evaluation.metrics import (
    METRIC_PRIORITY,
    PRIMARY_METRIC,
    PRIMARY_METRIC_DISPLAY_NAME,
)
from ml.src.evaluation.plots import plot_scenario_comparison
from ml.src.models.experiment_config import (
    DEFAULT_EXPERIMENTS_REPORTS_DIR,
    SCENARIO_REGION_EXCLUDED,
    SCENARIO_REGION_INCLUDED,
    to_relative_path,
)

logger = logging.getLogger(__name__)


def load_experiment_summary(summary_path: Path) -> dict[str, Any]:
    """Load an `experiment_summary.json` artifact."""
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment summary not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_comparable_summaries(
    included_summary: dict[str, Any],
    excluded_summary: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]], list[tuple[str, str]]]:
    """
    Strictly validate that two experiment summaries are directly comparable.

    Raises `ValueError` on any mismatch in scenario names, dataset fingerprint,
    row counts, CV/threshold configuration, target/model lists, `(target, model)`
    pair completeness, or target class distributions.
    """
    inc_cfg = included_summary.get("config", {})
    exc_cfg = excluded_summary.get("config", {})

    # 1. Expected scenario names
    if inc_cfg.get("scenario") != SCENARIO_REGION_INCLUDED:
        raise ValueError(
            f"Expected first summary scenario to be '{SCENARIO_REGION_INCLUDED}', "
            f"got {inc_cfg.get('scenario')!r}."
        )
    if exc_cfg.get("scenario") != SCENARIO_REGION_EXCLUDED:
        raise ValueError(
            f"Expected second summary scenario to be '{SCENARIO_REGION_EXCLUDED}', "
            f"got {exc_cfg.get('scenario')!r}."
        )

    # 2. Dataset SHA-256 fingerprint
    inc_fp = included_summary.get("dataset_fingerprint")
    exc_fp = excluded_summary.get("dataset_fingerprint")
    if not inc_fp or not exc_fp:
        raise ValueError(
            "Both scenario summaries must include a non-empty 'dataset_fingerprint' "
            "to guarantee identical dataset content."
        )
    if inc_fp != exc_fp:
        raise ValueError(
            f"Dataset fingerprint mismatch between scenarios: '{inc_fp}' vs '{exc_fp}'."
        )

    # 3. Dataset row count
    if included_summary.get("dataset_row_count") != excluded_summary.get("dataset_row_count"):
        raise ValueError(
            "Dataset row count mismatch between scenarios: "
            f"{included_summary.get('dataset_row_count')} vs {excluded_summary.get('dataset_row_count')}."
        )

    # 4. CV and decision threshold settings
    if not inc_cfg.get("cv_strategy") or inc_cfg.get("cv_strategy") != exc_cfg.get("cv_strategy"):
        raise ValueError(
            f"Cannot fairly compare scenarios with different 'cv_strategy' settings: "
            f"{inc_cfg.get('cv_strategy')!r} vs {exc_cfg.get('cv_strategy')!r}."
        )

    for cfg_key in ("n_splits", "shuffle", "random_state", "decision_threshold"):
        if inc_cfg.get(cfg_key) != exc_cfg.get(cfg_key):
            raise ValueError(
                f"Cannot fairly compare scenarios with different '{cfg_key}' settings: "
                f"{inc_cfg.get(cfg_key)!r} vs {exc_cfg.get(cfg_key)!r}."
            )

    inc_thresh_opt = inc_cfg.get("threshold_optimized")
    exc_thresh_opt = exc_cfg.get("threshold_optimized")
    if inc_thresh_opt != exc_thresh_opt:
        raise ValueError(
            f"Cannot fairly compare scenarios with different 'threshold_optimized' settings: "
            f"{inc_thresh_opt!r} vs {exc_thresh_opt!r}."
        )
    if inc_thresh_opt is not False:
        raise ValueError(
            "Baseline scenario comparison requires 'threshold_optimized=False' in both scenarios, "
            f"got {inc_thresh_opt!r}."
        )

    # 5. Scenario feature list parity (only `region` may differ; `region_missing_flag` must be absent)
    inc_selected = list(inc_cfg.get("selected_features", []))
    exc_selected = list(exc_cfg.get("selected_features", []))
    inc_cat = list(inc_cfg.get("categorical_features", []))
    exc_cat = list(exc_cfg.get("categorical_features", []))
    inc_num = list(inc_cfg.get("numeric_features", []))
    exc_num = list(exc_cfg.get("numeric_features", []))

    for scenario_name, feat_lists in (
        (SCENARIO_REGION_INCLUDED, (inc_selected, inc_cat, inc_num)),
        (SCENARIO_REGION_EXCLUDED, (exc_selected, exc_cat, exc_num)),
    ):
        if any("region_missing_flag" in fl for fl in feat_lists):
            raise ValueError(
                f"Scenario '{scenario_name}' must not include 'region_missing_flag' in feature lists."
            )

    if "region" not in inc_selected or "region" not in inc_cat:
        raise ValueError(
            f"Scenario '{SCENARIO_REGION_INCLUDED}' must include 'region' in both "
            "'selected_features' and 'categorical_features'."
        )
    if "region" in inc_num:
        raise ValueError(
            f"Scenario '{SCENARIO_REGION_INCLUDED}' must not include 'region' in 'numeric_features'."
        )
    if "region" in exc_selected or "region" in exc_cat or "region" in exc_num:
        raise ValueError(
            f"Scenario '{SCENARIO_REGION_EXCLUDED}' must not include 'region' in any feature list."
        )

    inc_selected_without_region = [f for f in inc_selected if f != "region"]
    if not exc_selected or inc_selected_without_region != exc_selected:
        raise ValueError(
            "Feature list mismatch between scenarios: removing 'region' from "
            f"'{SCENARIO_REGION_INCLUDED}' selected_features ({inc_selected_without_region}) "
            f"must match '{SCENARIO_REGION_EXCLUDED}' selected_features ({exc_selected})."
        )

    inc_cat_without_region = [f for f in inc_cat if f != "region"]
    if inc_cat_without_region != exc_cat:
        raise ValueError(
            "Categorical feature list mismatch between scenarios: removing 'region' from "
            f"'{SCENARIO_REGION_INCLUDED}' categorical_features ({inc_cat_without_region}) "
            f"must match '{SCENARIO_REGION_EXCLUDED}' categorical_features ({exc_cat})."
        )

    if inc_num != exc_num:
        raise ValueError(
            f"Numeric feature list mismatch between scenarios: {inc_num} vs {exc_num}."
        )

    # 6. Exact target and model lists
    inc_targets = list(inc_cfg.get("targets", []))
    exc_targets = list(exc_cfg.get("targets", []))
    if not inc_targets or inc_targets != exc_targets:
        raise ValueError(
            f"Target list mismatch between scenarios: {inc_targets} vs {exc_targets}."
        )

    inc_models = list(inc_cfg.get("models", []))
    exc_models = list(exc_cfg.get("models", []))
    if not inc_models or inc_models != exc_models:
        raise ValueError(
            f"Model list mismatch between scenarios: {inc_models} vs {exc_models}."
        )

    # 6. Complete (target, model) pair coverage without silently dropping missing pairs
    inc_index: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in included_summary.get("results", []):
        key = (entry["target"], entry["model"])
        if key in inc_index:
            raise ValueError(f"Duplicate (target, model) entry in {SCENARIO_REGION_INCLUDED}: {key}")
        inc_index[key] = entry

    exc_index: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in excluded_summary.get("results", []):
        key = (entry["target"], entry["model"])
        if key in exc_index:
            raise ValueError(f"Duplicate (target, model) entry in {SCENARIO_REGION_EXCLUDED}: {key}")
        exc_index[key] = entry

    expected_keys = {(t, m) for t in inc_targets for m in inc_models}
    missing_in_inc = sorted(expected_keys - set(inc_index.keys()))
    missing_in_exc = sorted(expected_keys - set(exc_index.keys()))
    extra_in_inc = sorted(set(inc_index.keys()) - expected_keys)
    extra_in_exc = sorted(set(exc_index.keys()) - expected_keys)

    if missing_in_inc or missing_in_exc or extra_in_inc or extra_in_exc:
        raise ValueError(
            "Incomplete or mismatched (target, model) experiment pairs between scenarios: "
            f"missing_in_{SCENARIO_REGION_INCLUDED}={missing_in_inc}, "
            f"missing_in_{SCENARIO_REGION_EXCLUDED}={missing_in_exc}, "
            f"extra_in_{SCENARIO_REGION_INCLUDED}={extra_in_inc}, "
            f"extra_in_{SCENARIO_REGION_EXCLUDED}={extra_in_exc}."
        )

    # 7. Target class distribution parity across both scenarios
    ordered_keys = sorted(expected_keys)
    for key in ordered_keys:
        inc_dist = inc_index[key]["class_distribution"]
        exc_dist = exc_index[key]["class_distribution"]
        for dist_key in ("n_samples", "n_positive", "n_negative"):
            if inc_dist.get(dist_key) != exc_dist.get(dist_key):
                raise ValueError(
                    f"Target class distribution mismatch for {key} on '{dist_key}': "
                    f"{inc_dist.get(dist_key)} vs {exc_dist.get(dist_key)}."
                )
        inc_rate = inc_dist.get("positive_rate_pct")
        exc_rate = exc_dist.get("positive_rate_pct")
        if inc_rate is None or exc_rate is None or not np.isclose(float(inc_rate), float(exc_rate)):
            raise ValueError(
                f"Target positive_rate_pct mismatch for {key}: {inc_rate} vs {exc_rate}."
            )

    return inc_index, exc_index, ordered_keys


def build_comparison_table(
    included_summary: dict[str, Any],
    excluded_summary: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a unified comparison DataFrame aligning all `(target, model)` pairs
    across `region_included` and `region_excluded` after strict parity checks.
    """
    inc_index, exc_index, ordered_keys = _validate_comparable_summaries(
        included_summary, excluded_summary
    )

    rows: list[dict[str, Any]] = []
    for target, model in ordered_keys:
        inc_entry = inc_index[(target, model)]
        exc_entry = exc_index[(target, model)]

        row: dict[str, Any] = {
            "target": target,
            "model": model,
            "n_samples": inc_entry["class_distribution"]["n_samples"],
            "n_positive": inc_entry["class_distribution"]["n_positive"],
            "positive_rate_pct": inc_entry["class_distribution"]["positive_rate_pct"],
        }

        for metric in METRIC_PRIORITY:
            inc_mean = float(inc_entry["summary_metrics"][metric]["mean"])
            inc_std = float(inc_entry["summary_metrics"][metric]["std"])
            exc_mean = float(exc_entry["summary_metrics"][metric]["mean"])
            exc_std = float(exc_entry["summary_metrics"][metric]["std"])
            row[f"{SCENARIO_REGION_INCLUDED}_{metric}_mean"] = inc_mean
            row[f"{SCENARIO_REGION_INCLUDED}_{metric}_std"] = inc_std
            row[f"{SCENARIO_REGION_EXCLUDED}_{metric}_mean"] = exc_mean
            row[f"{SCENARIO_REGION_EXCLUDED}_{metric}_std"] = exc_std
            row[f"delta_{metric}_mean"] = inc_mean - exc_mean

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["target", f"{SCENARIO_REGION_INCLUDED}_{PRIMARY_METRIC}_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return df


def render_comparison_markdown(
    comparison_df: pd.DataFrame,
    included_summary: dict[str, Any],
    excluded_summary: dict[str, Any],
) -> str:
    """Render a human-readable Markdown report comparing the two region scenarios."""
    lines: list[str] = [
        "# Baseline Experiment Comparison: Region Included vs Region Excluded",
        "",
        "## Comparison Design & Methodology",
        f"- **Dataset Row Count**: {included_summary.get('dataset_row_count')}",
        f"- **Dataset Fingerprint (SHA-256)**: `{included_summary.get('dataset_fingerprint')}`",
        f"- **Cross-Validation**: {included_summary['config'].get('cv_strategy')} "
        f"(n_splits={included_summary['config'].get('n_splits')}, "
        f"shuffle={included_summary['config'].get('shuffle')}, "
        f"random_state={included_summary['config'].get('random_state')})",
        f"- **Primary Comparison Metric**: `{PRIMARY_METRIC}` — {PRIMARY_METRIC_DISPLAY_NAME}",
        f"- **Decision Threshold**: `{included_summary['config'].get('decision_threshold')}` "
        "(**unoptimized** baseline threshold; used for Recall, Precision, F1)",
        "- **Model Selection Policy**: Accuracy is intentionally excluded due to class imbalance.",
        f"- **Region Included Features ({len(included_summary['config']['selected_features'])})**: "
        + ", ".join(f"`{c}`" for c in included_summary["config"]["selected_features"]),
        f"- **Region Excluded Features ({len(excluded_summary['config']['selected_features'])})**: "
        + ", ".join(f"`{c}`" for c in excluded_summary["config"]["selected_features"]),
        "",
    ]

    for target, group in comparison_df.groupby("target", sort=True):
        first = group.iloc[0]
        lines.extend(
            [
                f"## Target: `{target}` (Positives: {int(first['n_positive'])} / "
                f"{int(first['n_samples'])}, {first['positive_rate_pct']:.2f}%)",
                "",
                "| Model | Average Precision (AP) (Included) | Average Precision (AP) (Excluded) | Δ AP | ROC-AUC (Inc) | ROC-AUC (Exc) | Brier (Inc) | Brier (Exc) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, r in group.iterrows():
            lines.append(
                f"| `{r['model']}` | "
                f"{r['region_included_average_precision_mean']:.4f} ± {r['region_included_average_precision_std']:.4f} | "
                f"{r['region_excluded_average_precision_mean']:.4f} ± {r['region_excluded_average_precision_std']:.4f} | "
                f"{r['delta_average_precision_mean']:+.4f} | "
                f"{r['region_included_roc_auc_mean']:.4f} | "
                f"{r['region_excluded_roc_auc_mean']:.4f} | "
                f"{r['region_included_brier_score_mean']:.4f} | "
                f"{r['region_excluded_brier_score_mean']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def compare_scenarios(
    reports_base_dir: Path = DEFAULT_EXPERIMENTS_REPORTS_DIR,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """
    Load summaries from `region_included/` and `region_excluded/`, produce
    comparison CSV, JSON, Markdown report, and optional bar charts under
    `comparison/`.
    """
    base_dir = Path(reports_base_dir)
    inc_path = base_dir / SCENARIO_REGION_INCLUDED / "experiment_summary.json"
    exc_path = base_dir / SCENARIO_REGION_EXCLUDED / "experiment_summary.json"
    out_dir = base_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    included_summary = load_experiment_summary(inc_path)
    excluded_summary = load_experiment_summary(exc_path)

    comparison_df = build_comparison_table(included_summary, excluded_summary)

    csv_path = out_dir / "comparison_summary.csv"
    comparison_df.to_csv(csv_path, index=False)

    plot_paths: list[str] = []
    if generate_plots:
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for target in comparison_df["target"].unique():
            fig_path = plot_scenario_comparison(
                comparison_df=comparison_df,
                target=str(target),
                metric=PRIMARY_METRIC,
                output_path=figures_dir / f"scenario_comparison_{target}_{PRIMARY_METRIC}.png",
            )
            plot_paths.append(to_relative_path(fig_path))

    report_md = render_comparison_markdown(comparison_df, included_summary, excluded_summary)
    md_path = out_dir / "comparison_report.md"
    md_path.write_text(report_md, encoding="utf-8")

    payload: dict[str, Any] = {
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_display_name": PRIMARY_METRIC_DISPLAY_NAME,
        "dataset_fingerprint": included_summary.get("dataset_fingerprint"),
        "decision_threshold": included_summary["config"].get("decision_threshold"),
        "threshold_optimized": False,
        "included_summary_path": to_relative_path(inc_path),
        "excluded_summary_path": to_relative_path(exc_path),
        "comparison_csv_path": to_relative_path(csv_path),
        "comparison_report_path": to_relative_path(md_path),
        "plot_paths": plot_paths,
        "records": comparison_df.to_dict(orient="records"),
    }

    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Scenario comparison written to %s", to_relative_path(out_dir))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare region_included vs region_excluded baseline experiments."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_REPORTS_DIR,
        help="Base experiments report directory containing region_included/ and region_excluded/.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating comparison charts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    compare_scenarios(
        reports_base_dir=args.reports_dir,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
