"""B2B Churn Intelligence — Model Evaluation & Scenario Comparison Package.

Uses PEP 562 lazy attribute loading (`__getattr__`) so submodules such as
`ml.src.evaluation.compare_experiments` can be executed directly with
`python -m ml.src.evaluation.compare_experiments` without triggering a `runpy`
`RuntimeWarning`, while preserving full public package exports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # metrics
    "DEFAULT_DECISION_THRESHOLD": ("ml.src.evaluation.metrics", "DEFAULT_DECISION_THRESHOLD"),
    "METRIC_HIGHER_IS_BETTER": ("ml.src.evaluation.metrics", "METRIC_HIGHER_IS_BETTER"),
    "METRIC_PRIORITY": ("ml.src.evaluation.metrics", "METRIC_PRIORITY"),
    "PRIMARY_METRIC": ("ml.src.evaluation.metrics", "PRIMARY_METRIC"),
    "PRIMARY_METRIC_DISPLAY_NAME": ("ml.src.evaluation.metrics", "PRIMARY_METRIC_DISPLAY_NAME"),
    "compute_binary_metrics": ("ml.src.evaluation.metrics", "compute_binary_metrics"),
    "compute_class_distribution": ("ml.src.evaluation.metrics", "compute_class_distribution"),
    "evaluate_oof_predictions": ("ml.src.evaluation.metrics", "evaluate_oof_predictions"),
    "summarize_fold_metrics": ("ml.src.evaluation.metrics", "summarize_fold_metrics"),
    "validate_binary_target": ("ml.src.evaluation.metrics", "validate_binary_target"),
    # plots
    "plot_pr_curves": ("ml.src.evaluation.plots", "plot_pr_curves"),
    "plot_roc_curves": ("ml.src.evaluation.plots", "plot_roc_curves"),
    "plot_scenario_comparison": ("ml.src.evaluation.plots", "plot_scenario_comparison"),
    # compare_experiments
    "build_comparison_table": ("ml.src.evaluation.compare_experiments", "build_comparison_table"),
    "compare_scenarios": ("ml.src.evaluation.compare_experiments", "compare_scenarios"),
    "load_experiment_summary": ("ml.src.evaluation.compare_experiments", "load_experiment_summary"),
    "render_comparison_markdown": (
        "ml.src.evaluation.compare_experiments",
        "render_comparison_markdown",
    ),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
