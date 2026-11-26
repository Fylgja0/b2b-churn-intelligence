"""
Centralized Binary Classification Evaluation Metrics.

Metric priority order (aligned with imbalanced churn modeling):
  1. average_precision — Average Precision (AP) (primary comparison metric)
  2. roc_auc — ROC-AUC
  3. recall — Recall at default threshold 0.5 (unoptimized in baseline phase)
  4. precision — Precision at default threshold 0.5 (unoptimized in baseline phase)
  5. f1 — F1-score at default threshold 0.5 (unoptimized in baseline phase)
  6. brier_score — Brier Score (lower is better)

Note: `average_precision_score` computes Average Precision (AP), defined as the
weighted mean of precisions achieved at each threshold with the increase in
recall from the previous threshold used as the weight. It is reported explicitly
as Average Precision (AP) and not treated as synonymous with trapezoidal PR-AUC.

Accuracy is intentionally excluded from model selection criteria because of
severe class imbalance (e.g., 4.51% positives in target_churn_90d).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC = "average_precision"
PRIMARY_METRIC_DISPLAY_NAME = "Average Precision (AP)"

METRIC_PRIORITY: tuple[str, ...] = (
    "average_precision",
    "roc_auc",
    "recall",
    "precision",
    "f1",
    "brier_score",
)

# True = higher is better, False = lower is better
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "average_precision": True,
    "roc_auc": True,
    "recall": True,
    "precision": True,
    "f1": True,
    "brier_score": False,
}

DEFAULT_DECISION_THRESHOLD = 0.5


def validate_binary_target(
    y: Sequence[Any] | np.ndarray | pd.Series,
    name: str = "y_true",
) -> np.ndarray:
    """
    Strictly validate a 1-D binary target vector BEFORE casting to `int`.

    Rejects:
      - Non-1D or empty inputs
      - Missing values (`NaN`, `None`, `pd.NA`)
      - Fractional floats (e.g., `0.5`, `1.2`), infinities, strings, booleans,
        or any value outside `{0, 1}`
    """
    raw_arr = np.asarray(y, dtype=object)
    if raw_arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array, got shape {raw_arr.shape}.")
    if len(raw_arr) == 0:
        raise ValueError(f"{name} must not be empty.")

    if bool(np.any(pd.isna(raw_arr))):
        raise ValueError(f"{name} contains missing/NaN values.")

    validated = np.empty(len(raw_arr), dtype=int)
    for idx, val in enumerate(raw_arr):
        if isinstance(val, (bool, np.bool_)):
            raise ValueError(
                f"{name} must contain integer binary values 0 or 1, not boolean ({val!r})."
            )
        if isinstance(val, (int, np.integer)):
            int_val = int(val)
            if int_val not in (0, 1):
                raise ValueError(
                    f"{name} must contain only binary values 0 or 1. Found {val!r} at index {idx}."
                )
            validated[idx] = int_val
        elif isinstance(val, (float, np.floating)):
            fval = float(val)
            if not np.isfinite(fval) or fval not in (0.0, 1.0):
                raise ValueError(
                    f"{name} must not contain fractional or non-binary values. "
                    f"Found {val!r} at index {idx}."
                )
            validated[idx] = int(fval)
        else:
            raise ValueError(
                f"{name} must contain numeric binary values 0 or 1. "
                f"Found unsupported value {val!r} ({type(val).__name__}) at index {idx}."
            )

    return validated


def compute_binary_metrics(
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, float]:
    """
    Compute prioritized binary classification metrics from true labels and
    predicted positive-class probabilities.

    Precision, Recall, and F1 are computed at `threshold` (0.5 by default,
    unoptimized in the baseline phase).
    """
    y_true_arr = validate_binary_target(y_true, name="y_true")
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_prob_arr.ndim != 1:
        raise ValueError("y_prob must be a 1-dimensional array.")
    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_prob)={len(y_prob_arr)}"
        )
    if np.any(~np.isfinite(y_prob_arr)) or np.any(y_prob_arr < 0.0) or np.any(y_prob_arr > 1.0):
        raise ValueError("y_prob must contain finite probabilities in [0, 1].")
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be strictly between 0 and 1, got {threshold}.")

    y_pred = (y_prob_arr >= threshold).astype(int)
    n_classes_present = len(np.unique(y_true_arr))

    if n_classes_present < 2:
        ap = float("nan")
        roc_auc = float("nan")
    else:
        ap = float(average_precision_score(y_true_arr, y_prob_arr))
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "brier_score": brier,
    }


def summarize_fold_metrics(
    fold_metrics: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Aggregate fold-level metric dictionaries into mean, std (sample std, ddof=1
    when n_valid_folds > 1), min, and max for every metric in `METRIC_PRIORITY`.
    """
    if not fold_metrics:
        raise ValueError("fold_metrics must not be empty.")

    summary: dict[str, dict[str, float]] = {}
    for metric in METRIC_PRIORITY:
        values = np.array([float(fm[metric]) for fm in fold_metrics], dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            summary[metric] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        else:
            ddof = 1 if len(valid) > 1 else 0
            summary[metric] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid, ddof=ddof)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            }
    return summary


def compute_class_distribution(
    y: Sequence[Any] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute binary class counts, positive rate, and negative-to-positive ratio."""
    y_arr = validate_binary_target(y, name="y")

    n_samples = int(len(y_arr))
    n_positive = int(np.sum(y_arr == 1))
    n_negative = int(n_samples - n_positive)
    positive_rate = float(n_positive / n_samples)
    imbalance_ratio = float(n_negative / n_positive) if n_positive > 0 else float("inf")

    return {
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": positive_rate,
        "positive_rate_pct": round(positive_rate * 100.0, 2),
        "imbalance_ratio": round(imbalance_ratio, 2) if np.isfinite(imbalance_ratio) else None,
    }


def evaluate_oof_predictions(
    oof_df: pd.DataFrame,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, Any]:
    """
    Evaluate an out-of-fold predictions DataFrame containing at least
    `y_true`, `y_probability`, and `fold`.
    """
    required = {"y_true", "y_probability", "fold"}
    missing = sorted(required - set(oof_df.columns))
    if missing:
        raise ValueError(f"OOF DataFrame is missing required columns: {missing}")

    overall = compute_binary_metrics(
        oof_df["y_true"].to_numpy(),
        oof_df["y_probability"].to_numpy(),
        threshold=threshold,
    )

    fold_results: list[dict[str, Any]] = []
    for fold_id, group in oof_df.groupby("fold", sort=True):
        m = compute_binary_metrics(
            group["y_true"].to_numpy(),
            group["y_probability"].to_numpy(),
            threshold=threshold,
        )
        fold_results.append({"fold": int(fold_id), **m})

    return {
        "decision_threshold": threshold,
        "threshold_optimized": False,
        "overall_oof_metrics": overall,
        "fold_metrics": fold_results,
        "summary_metrics": summarize_fold_metrics(fold_results),
    }
