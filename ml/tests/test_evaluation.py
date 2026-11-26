"""
Unit and Integration Tests for Evaluation Metrics, Plots, and Scenario Comparison.

Covers:
  - Exact hand-verified binary classification metrics (Average Precision, ROC-AUC,
    Recall, Precision, F1 at threshold 0.5, Brier Score).
  - Edge cases: all-zero predictions (e.g. DummyClassifier), single-class validation,
    invalid probability bounds, length mismatches.
  - Fold metric aggregation (mean, std, min, max).
  - Class distribution and imbalance ratio reporting.
  - OOF DataFrame evaluation.
  - Scenario comparison (`region_included` vs `region_excluded`) table, delta
    computation, mismatch guards, Markdown rendering, and chart generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src.evaluation.compare_experiments import (
    build_comparison_table,
    compare_scenarios,
)
from ml.src.evaluation.metrics import (
    METRIC_PRIORITY,
    PRIMARY_METRIC,
    compute_binary_metrics,
    compute_class_distribution,
    evaluate_oof_predictions,
    summarize_fold_metrics,
)
from ml.src.evaluation.plots import plot_pr_curves, plot_roc_curves
from ml.src.models.experiment_config import (
    PRIMARY_TARGET,
    SCENARIO_REGION_EXCLUDED,
    SCENARIO_REGION_INCLUDED,
    build_experiment_config,
)
from ml.src.models.train import run_experiment
from ml.tests.test_models import make_synthetic_modeling_df


def test_compute_binary_metrics_exact_values() -> None:
    """13. Metric calculations match analytical ground truth at threshold 0.5."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.1])

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert list(metrics.keys()) == list(METRIC_PRIORITY)
    assert PRIMARY_METRIC == "average_precision"

    # At threshold 0.5: y_pred = [1, 0, 1, 0] -> TP=1, FP=1, FN=1, TN=1
    assert np.isclose(metrics["recall"], 0.5)
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["f1"], 0.5)

    # Brier = mean([(0.9-1)^2, (0.4-1)^2, (0.6-0)^2, (0.1-0)^2])
    expected_brier = (0.01 + 0.36 + 0.36 + 0.01) / 4.0
    assert np.isclose(metrics["brier_score"], expected_brier)

    # Ranks: 0.9(1), 0.6(0), 0.4(1), 0.1(0) -> ROC-AUC = 0.75
    assert np.isclose(metrics["roc_auc"], 0.75)
    # AP = (1/1 + 2/3) / 2 = 5/6
    assert np.isclose(metrics["average_precision"], 5.0 / 6.0)


def test_compute_binary_metrics_all_zero_predictions_safe() -> None:
    """When a model predicts all 0s at 0.5 (like Dummy prior on imbalanced data), metrics do not fail."""
    y_true = np.array([0, 0, 0, 0, 1])
    y_prob = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
    assert np.isclose(metrics["average_precision"], 0.2)
    assert np.isclose(metrics["roc_auc"], 0.5)
    assert np.isclose(metrics["brier_score"], 0.16)


def test_compute_binary_metrics_input_validation() -> None:
    """Invalid probabilities, non-binary labels, or mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_binary_metrics([0, 1], [0.5])

    with pytest.raises(ValueError, match="binary"):
        compute_binary_metrics([0, 2], [0.2, 0.8])

    with pytest.raises(ValueError, match="finite probabilities"):
        compute_binary_metrics([0, 1], [0.2, 1.5])


def test_summarize_fold_metrics_mean_and_std() -> None:
    """Fold summary accurately computes mean, sample std (ddof=1), min, and max."""
    fold_metrics = [
        {
            "fold": 0,
            "average_precision": 0.20,
            "roc_auc": 0.60,
            "recall": 0.10,
            "precision": 0.40,
            "f1": 0.16,
            "brier_score": 0.12,
        },
        {
            "fold": 1,
            "average_precision": 0.40,
            "roc_auc": 0.80,
            "recall": 0.30,
            "precision": 0.60,
            "f1": 0.40,
            "brier_score": 0.10,
        },
    ]
    summary = summarize_fold_metrics(fold_metrics)

    assert np.isclose(summary["average_precision"]["mean"], 0.30)
    assert np.isclose(summary["average_precision"]["std"], np.std([0.20, 0.40], ddof=1))
    assert np.isclose(summary["roc_auc"]["mean"], 0.70)
    assert np.isclose(summary["brier_score"]["min"], 0.10)
    assert np.isclose(summary["brier_score"]["max"], 0.12)


def test_compute_class_distribution_and_imbalance_ratio() -> None:
    """14. Class distribution reports exact sample counts, rates, and imbalance ratio."""
    # Replicates 90d-like class imbalance: 56 positives out of 1242
    y = np.array([1] * 56 + [0] * (1242 - 56))
    dist = compute_class_distribution(y)

    assert dist["n_samples"] == 1242
    assert dist["n_positive"] == 56
    assert dist["n_negative"] == 1186
    assert dist["positive_rate_pct"] == 4.51
    assert dist["imbalance_ratio"] == round(1186 / 56, 2)


def test_evaluate_oof_predictions_and_plotting(tmp_path: Path) -> None:
    """OOF evaluation and PR/ROC curve plots run cleanly on synthetic OOF output."""
    oof_df = pd.DataFrame(
        {
            "subscription_id": [f"S{i:03d}" for i in range(8)],
            "account_id": [f"A{i:03d}" for i in range(8)],
            "target": PRIMARY_TARGET,
            "model": "logistic_regression",
            "y_true": [1, 0, 1, 0, 1, 0, 1, 0],
            "y_probability": [0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6, 0.1],
            "y_pred_0_5": [1, 0, 1, 0, 1, 0, 1, 0],
            "fold": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    result = evaluate_oof_predictions(oof_df, threshold=0.5)
    assert result["threshold_optimized"] is False
    assert np.isclose(result["overall_oof_metrics"]["roc_auc"], 1.0)
    assert np.isclose(result["overall_oof_metrics"]["average_precision"], 1.0)
    assert len(result["fold_metrics"]) == 2

    pr_path = plot_pr_curves({"logistic_regression": oof_df}, PRIMARY_TARGET, tmp_path / "pr.png")
    roc_path = plot_roc_curves({"logistic_regression": oof_df}, PRIMARY_TARGET, tmp_path / "roc.png")
    assert pr_path.exists() and pr_path.stat().st_size > 0
    assert roc_path.exists() and roc_path.stat().st_size > 0


def test_compare_scenarios_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    """Scenario comparison aligns both scenarios, computes deltas, and generates reports."""
    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    csv_path = tmp_path / "synthetic_modeling.csv"
    df.to_csv(csv_path, index=False)

    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"

    for scenario in (SCENARIO_REGION_INCLUDED, SCENARIO_REGION_EXCLUDED):
        cfg = build_experiment_config(
            scenario=scenario,
            targets=[PRIMARY_TARGET],
            models=["dummy", "logistic_regression"],
            dataset_path=csv_path,
            artifacts_base_dir=artifacts_dir,
            reports_base_dir=reports_dir,
        )
        run_experiment(cfg, save_artifacts=False, generate_plots=False)

    comparison = compare_scenarios(reports_base_dir=reports_dir, generate_plots=True)
    assert comparison["primary_metric"] == PRIMARY_METRIC
    assert comparison["threshold_optimized"] is False
    assert len(comparison["records"]) == 2

    comp_dir = reports_dir / "comparison"
    assert (comp_dir / "comparison_summary.csv").exists()
    assert (comp_dir / "comparison_summary.json").exists()
    assert (comp_dir / "comparison_report.md").exists()
    assert (comp_dir / "figures" / f"scenario_comparison_{PRIMARY_TARGET}_{PRIMARY_METRIC}.png").exists()

    # Dummy prior model depends only on y, so delta between scenarios must be 0.0
    dummy_rec = next(r for r in comparison["records"] if r["model"] == "dummy")
    assert np.isclose(dummy_rec["delta_average_precision_mean"], 0.0)


def _make_valid_summary_pair() -> tuple[dict, dict]:
    """Helper producing a minimal valid pair of comparable scenario summaries."""
    base_metrics = {
        m: {"mean": 0.5, "std": 0.05, "min": 0.45, "max": 0.55}
        for m in METRIC_PRIORITY
    }
    base_dist = {
        "n_samples": 100,
        "n_positive": 20,
        "n_negative": 80,
        "positive_rate": 0.2,
        "positive_rate_pct": 20.0,
        "imbalance_ratio": 4.0,
    }

    def _build(scenario: str) -> dict:
        cat_features = (
            ["region", "segment", "acquisition_channel", "plan_name"]
            if scenario == SCENARIO_REGION_INCLUDED
            else ["segment", "acquisition_channel", "plan_name"]
        )
        num_features = [
            "account_age_days_at_t0",
            "account_created_at_missing_flag",
            "subscription_tenure_days_at_t0",
            "subscription_start_month",
            "subscription_start_quarter",
            "plan_monthly_price",
            "plan_included_seats",
        ]
        selected_features = (
            ["region", "segment", "acquisition_channel", *num_features[:5], "plan_name", *num_features[5:]]
            if scenario == SCENARIO_REGION_INCLUDED
            else ["segment", "acquisition_channel", *num_features[:5], "plan_name", *num_features[5:]]
        )
        return {
            "dataset_row_count": 100,
            "dataset_fingerprint": "a" * 64,
            "config": {
                "scenario": scenario,
                "targets": [PRIMARY_TARGET],
                "models": ["dummy", "logistic_regression"],
                "selected_features": selected_features,
                "categorical_features": cat_features,
                "numeric_features": num_features,
                "cv_strategy": "StratifiedKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
                "decision_threshold": 0.5,
                "threshold_optimized": False,
            },
            "results": [
                {
                    "scenario": scenario,
                    "target": PRIMARY_TARGET,
                    "model": "dummy",
                    "class_distribution": dict(base_dist),
                    "summary_metrics": dict(base_metrics),
                },
                {
                    "scenario": scenario,
                    "target": PRIMARY_TARGET,
                    "model": "logistic_regression",
                    "class_distribution": dict(base_dist),
                    "summary_metrics": dict(base_metrics),
                },
            ],
        }

    return _build(SCENARIO_REGION_INCLUDED), _build(SCENARIO_REGION_EXCLUDED)


def test_build_comparison_table_rejects_mismatched_cv_config() -> None:
    """Fair comparison guard rejects mismatched row counts, CV settings, and thresholds."""
    inc, exc = _make_valid_summary_pair()
    exc["dataset_row_count"] = 90
    with pytest.raises(ValueError, match="row count mismatch"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["cv_strategy"] = "GroupKFold"
    with pytest.raises(ValueError, match="cv_strategy"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["n_splits"] = 3
    with pytest.raises(ValueError, match="n_splits"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["shuffle"] = False
    with pytest.raises(ValueError, match="shuffle"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["random_state"] = 99
    with pytest.raises(ValueError, match="random_state"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["decision_threshold"] = 0.3
    with pytest.raises(ValueError, match="decision_threshold"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["threshold_optimized"] = True
    with pytest.raises(ValueError, match="different 'threshold_optimized'"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    inc["config"]["threshold_optimized"] = True
    exc["config"]["threshold_optimized"] = True
    with pytest.raises(ValueError, match="threshold_optimized=False"):
        build_comparison_table(inc, exc)


def test_build_comparison_table_rejects_fingerprint_and_scenario_mismatches() -> None:
    """Different dataset fingerprints or unexpected scenario names are rejected."""
    inc, exc = _make_valid_summary_pair()
    exc["dataset_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    del exc["dataset_fingerprint"]
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    inc["config"]["scenario"] = "region_excluded"
    with pytest.raises(ValueError, match="Expected first summary scenario"):
        build_comparison_table(inc, exc)


def test_build_comparison_table_rejects_target_model_pair_and_distribution_mismatches() -> None:
    """Mismatched target/model lists, missing pairs, or class distribution drift fail loudly."""
    # Different target lists
    inc, exc = _make_valid_summary_pair()
    exc["config"]["targets"] = ["target_churn_180d"]
    with pytest.raises(ValueError, match="Target list mismatch"):
        build_comparison_table(inc, exc)

    # Different model lists
    inc, exc = _make_valid_summary_pair()
    exc["config"]["models"] = ["dummy", "random_forest"]
    with pytest.raises(ValueError, match="Model list mismatch"):
        build_comparison_table(inc, exc)

    # Missing (target, model) experiment pair must NOT be silently skipped
    inc, exc = _make_valid_summary_pair()
    exc["results"] = [r for r in exc["results"] if r["model"] != "logistic_regression"]
    with pytest.raises(ValueError, match="Incomplete or mismatched"):
        build_comparison_table(inc, exc)

    # Different target class distributions
    inc, exc = _make_valid_summary_pair()
    exc["results"][0]["class_distribution"]["n_positive"] = 25
    with pytest.raises(ValueError, match="class distribution mismatch"):
        build_comparison_table(inc, exc)


def test_binary_target_validation_rejects_fractional_nan_and_non_binary() -> None:
    """Fractional, NaN, None, or non-binary targets are rejected without silent int truncation."""
    bad_targets = [
        [0.0, 0.5, 1.0],
        [0.0, 1.2, 0.0],
        [0, np.nan, 1],
        [0, None, 1],
        [0, -1, 1],
        [0, 2, 1],
        ["0", "1", "0"],
    ]
    probs = [0.2, 0.5, 0.8]

    for bad_y in bad_targets:
        with pytest.raises(ValueError):
            compute_binary_metrics(bad_y, probs)
        with pytest.raises(ValueError):
            compute_class_distribution(bad_y)


def test_single_class_validation_fold_metrics_handling() -> None:
    """Single-class validation fold yields NaN for AP/ROC-AUC while keeping Brier/F1 valid."""
    single_class_metrics = compute_binary_metrics(
        y_true=[0, 0, 0, 0],
        y_prob=[0.1, 0.2, 0.3, 0.4],
        threshold=0.5,
    )
    assert np.isnan(single_class_metrics["average_precision"])
    assert np.isnan(single_class_metrics["roc_auc"])
    assert np.isfinite(single_class_metrics["brier_score"])
    assert single_class_metrics["recall"] == 0.0
    assert single_class_metrics["precision"] == 0.0
    assert single_class_metrics["f1"] == 0.0

    # Summary across one valid fold + one single-class fold ignores NaN cleanly
    valid_fold = compute_binary_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], threshold=0.5)
    summary = summarize_fold_metrics([single_class_metrics, valid_fold])
    assert np.isclose(summary["average_precision"]["mean"], 1.0)
    assert np.isclose(summary["roc_auc"]["mean"], 1.0)
    assert np.isfinite(summary["brier_score"]["mean"])


def test_evaluation_package_lazy_exports_and_ap_naming() -> None:
    """Lazy exports in ml.src.evaluation work cleanly and AP naming is explicit."""
    import ml.src.evaluation as evaluation_pkg

    assert evaluation_pkg.PRIMARY_METRIC == "average_precision"
    assert evaluation_pkg.PRIMARY_METRIC_DISPLAY_NAME == "Average Precision (AP)"
    assert callable(evaluation_pkg.compare_scenarios)
    assert callable(evaluation_pkg.compute_binary_metrics)


def test_build_comparison_table_rejects_unexpected_feature_differences() -> None:
    """Only `region` may differ between scenarios; any other feature or order diff fails."""
    # 1. `region` missing in region_included
    inc, exc = _make_valid_summary_pair()
    inc["config"]["selected_features"] = [f for f in inc["config"]["selected_features"] if f != "region"]
    inc["config"]["categorical_features"] = [f for f in inc["config"]["categorical_features"] if f != "region"]
    with pytest.raises(ValueError, match="must include 'region'"):
        build_comparison_table(inc, exc)

    # 2. `region` unexpectedly present in region_excluded
    inc, exc = _make_valid_summary_pair()
    exc["config"]["selected_features"] = ["region", *exc["config"]["selected_features"]]
    exc["config"]["categorical_features"] = ["region", *exc["config"]["categorical_features"]]
    with pytest.raises(ValueError, match="must not include 'region'"):
        build_comparison_table(inc, exc)

    # 3. Different feature order in region_excluded selected_features
    inc, exc = _make_valid_summary_pair()
    exc["config"]["selected_features"] = list(reversed(exc["config"]["selected_features"]))
    with pytest.raises(ValueError, match="Feature list mismatch"):
        build_comparison_table(inc, exc)

    # 4. Different categorical_features (beyond `region`)
    inc, exc = _make_valid_summary_pair()
    exc["config"]["categorical_features"] = ["plan_name", "segment", "acquisition_channel"]
    with pytest.raises(ValueError, match="Categorical feature list mismatch"):
        build_comparison_table(inc, exc)

    # 5. Different numeric_features
    inc, exc = _make_valid_summary_pair()
    exc["config"]["numeric_features"] = exc["config"]["numeric_features"][:-1]
    with pytest.raises(ValueError, match="Numeric feature list mismatch"):
        build_comparison_table(inc, exc)


def test_build_comparison_table_rejects_region_missing_flag() -> None:
    """`region_missing_flag` must not appear in either scenario's feature lists."""
    inc, exc = _make_valid_summary_pair()
    inc["config"]["selected_features"] = [*inc["config"]["selected_features"], "region_missing_flag"]
    inc["config"]["numeric_features"] = [*inc["config"]["numeric_features"], "region_missing_flag"]
    with pytest.raises(ValueError, match="region_missing_flag"):
        build_comparison_table(inc, exc)

    inc, exc = _make_valid_summary_pair()
    exc["config"]["selected_features"] = [*exc["config"]["selected_features"], "region_missing_flag"]
    exc["config"]["numeric_features"] = [*exc["config"]["numeric_features"], "region_missing_flag"]
    with pytest.raises(ValueError, match="region_missing_flag"):
        build_comparison_table(inc, exc)
