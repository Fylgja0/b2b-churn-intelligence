"""
Deterministic Stratified Cross-Validation & Out-of-Fold (OOF) Prediction Engine.

Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` so that for a
given target variable, both `region_included` and `region_excluded` scenarios
evaluate on the exact same fold assignments.

Key guarantees:
  - Target input is strictly validated (1-D, no NaNs, binary 0/1, both classes
    have at least `n_splits` samples).
  - Custom split lists are validated BEFORE training begins (index bounds,
    disjoint train/val indices, non-empty folds, exact-once validation coverage).
  - Positional index alignment between `X`, `y`, and `metadata_df` is enforced.
  - Every pipeline is cloned fresh per fold and fit exclusively on the training
    fold (`X.iloc[train_idx]`, `y[train_idx]`) — zero validation leakage.
  - Every observation receives an OOF prediction from its own validation fold
    exactly once.
  - ID columns (`subscription_id`, `account_id`) are never passed to the model;
    they are attached solely to the OOF output table for record traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.src.evaluation.metrics import (
    DEFAULT_DECISION_THRESHOLD,
    compute_binary_metrics,
    compute_class_distribution,
    summarize_fold_metrics,
    validate_binary_target,
)
from ml.src.models.experiment_config import DEFAULT_CV_SPLITS, RANDOM_STATE

OOF_REQUIRED_COLUMNS: tuple[str, ...] = (
    "subscription_id",
    "account_id",
    "target",
    "model",
    "y_true",
    "y_probability",
    "y_pred_0_5",
    "fold",
)


def generate_cv_splits(
    y: Sequence[Any] | pd.Series | np.ndarray,
    n_splits: int = DEFAULT_CV_SPLITS,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate deterministic `StratifiedKFold` train/validation index splits.

    Validates that `y` is 1-D, contains no missing values, is strictly binary
    (0/1), and has at least `n_splits` samples in both classes.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    y_arr = validate_binary_target(y, name="y")

    if len(y_arr) < n_splits:
        raise ValueError(f"Number of samples ({len(y_arr)}) is less than n_splits ({n_splits}).")

    n_positive = int(np.sum(y_arr == 1))
    n_negative = int(np.sum(y_arr == 0))
    if n_positive < n_splits or n_negative < n_splits:
        raise ValueError(
            f"Insufficient samples per class for {n_splits}-fold StratifiedKFold: "
            f"n_positive={n_positive}, n_negative={n_negative} "
            f"(each class requires at least {n_splits} samples)."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    placeholder_x = np.zeros(len(y_arr), dtype=np.int8)
    return [(train_idx, val_idx) for train_idx, val_idx in skf.split(placeholder_x, y_arr)]


def validate_cv_splits(
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    n_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Validate a sequence of `(train_idx, val_idx)` splits before model training.

    Enforces:
      - Non-empty splits sequence
      - Integer 1-D index arrays strictly within `[0, n_samples - 1]`
      - Non-empty train and validation partitions without internal duplicates
      - Zero overlap between train and validation indices in every fold
      - Every sample `0..n_samples-1` appears in validation folds exactly once
    """
    if not splits:
        raise ValueError("CV splits must not be empty.")

    validated: list[tuple[np.ndarray, np.ndarray]] = []
    val_counts = np.zeros(n_samples, dtype=int)

    for fold_idx, pair in enumerate(splits):
        if len(pair) != 2:
            raise ValueError(f"Fold {fold_idx} must be a (train_idx, val_idx) tuple.")
        train_raw, val_raw = pair
        train_idx = np.asarray(train_raw)
        val_idx = np.asarray(val_raw)

        if train_idx.ndim != 1 or val_idx.ndim != 1:
            raise ValueError(f"Fold {fold_idx} indices must be 1-dimensional arrays.")
        if not np.issubdtype(train_idx.dtype, np.integer) or not np.issubdtype(
            val_idx.dtype, np.integer
        ):
            raise ValueError(f"Fold {fold_idx} indices must have integer dtype.")
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold_idx} has an empty train or validation split.")

        if np.any(train_idx < 0) or np.any(train_idx >= n_samples):
            raise ValueError(
                f"Fold {fold_idx} train_idx contains out-of-range indices for n_samples={n_samples}."
            )
        if np.any(val_idx < 0) or np.any(val_idx >= n_samples):
            raise ValueError(
                f"Fold {fold_idx} val_idx contains out-of-range indices for n_samples={n_samples}."
            )

        if len(np.unique(train_idx)) != len(train_idx):
            raise ValueError(f"Fold {fold_idx} train_idx contains duplicate indices.")
        if len(np.unique(val_idx)) != len(val_idx):
            raise ValueError(f"Fold {fold_idx} val_idx contains duplicate indices.")

        overlap = np.intersect1d(train_idx, val_idx)
        if len(overlap) > 0:
            raise ValueError(
                f"Fold {fold_idx} has overlapping train and validation indices "
                f"(count={len(overlap)}, sample={overlap[:5].tolist()})."
            )

        val_counts[val_idx] += 1
        validated.append((train_idx.astype(int, copy=False), val_idx.astype(int, copy=False)))

    if not np.all(val_counts == 1):
        missing_count = int(np.sum(val_counts == 0))
        duplicate_count = int(np.sum(val_counts > 1))
        raise ValueError(
            "Invalid CV splits: every observation must appear in validation folds exactly once "
            f"(missing={missing_count}, duplicated={duplicate_count})."
        )

    return validated


def summarize_fold_class_distributions(
    y: Sequence[Any] | pd.Series | np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Report train and validation sample counts and positive counts per fold."""
    y_arr = validate_binary_target(y, name="y")
    reports: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_dist = compute_class_distribution(y_arr[train_idx])
        val_dist = compute_class_distribution(y_arr[val_idx])
        reports.append(
            {
                "fold": fold_idx,
                "train_n_samples": train_dist["n_samples"],
                "train_n_positive": train_dist["n_positive"],
                "train_n_negative": train_dist["n_negative"],
                "train_positive_rate_pct": train_dist["positive_rate_pct"],
                "val_n_samples": val_dist["n_samples"],
                "val_n_positive": val_dist["n_positive"],
                "val_n_negative": val_dist["n_negative"],
                "val_positive_rate_pct": val_dist["positive_rate_pct"],
            }
        )
    return reports


@dataclass
class CVResult:
    """Structured output of cross-validation for one (scenario, target, model)."""

    scenario: str
    target: str
    model_name: str
    decision_threshold: float
    threshold_optimized: bool
    class_distribution: dict[str, Any]
    fold_distributions: list[dict[str, Any]]
    fold_metrics: list[dict[str, Any]]
    summary_metrics: dict[str, dict[str, float]]
    oof_metrics: dict[str, float]
    oof_predictions: pd.DataFrame
    fitted_fold_pipelines: list[Pipeline] = field(default_factory=list, repr=False)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize evaluation results (excluding DataFrame/pipelines) for JSON reports."""
        return {
            "scenario": self.scenario,
            "target": self.target,
            "model": self.model_name,
            "decision_threshold": self.decision_threshold,
            "threshold_optimized": self.threshold_optimized,
            "class_distribution": self.class_distribution,
            "fold_distributions": self.fold_distributions,
            "fold_metrics": self.fold_metrics,
            "summary_metrics": self.summary_metrics,
            "oof_metrics": self.oof_metrics,
        }


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Sequence[Any] | pd.Series | np.ndarray,
    metadata_df: pd.DataFrame,
    scenario: str,
    target: str,
    model_name: str,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    n_splits: int = DEFAULT_CV_SPLITS,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    retain_fold_pipelines: bool = False,
) -> CVResult:
    """
    Execute stratified K-fold cross-validation and collect OOF predictions.

    Parameters
    ----------
    pipeline:
        Unfitted scikit-learn `Pipeline` containing preprocessing and classifier.
    X:
        Feature matrix containing ONLY scenario-approved feature columns.
    y:
        Binary target vector aligned row-for-row with `X`.
    metadata_df:
        DataFrame with `subscription_id` and `account_id` aligned with `X`.
        Used ONLY to tag OOF outputs; never passed to `pipeline`.
    splits:
        Optional pre-computed `(train_idx, val_idx)` splits so multiple models
        and scenarios share the exact same fold assignments. Validated strictly
        before any model fitting starts.
    """
    required_meta = {"subscription_id", "account_id"}
    missing_meta = sorted(required_meta - set(metadata_df.columns))
    if missing_meta:
        raise ValueError(f"metadata_df is missing required ID columns: {missing_meta}")

    n_samples = len(X)
    y_arr = validate_binary_target(y, name="y")
    if len(y_arr) != n_samples or len(metadata_df) != n_samples:
        raise ValueError(
            f"Row count mismatch: X={n_samples}, y={len(y_arr)}, metadata={len(metadata_df)}"
        )

    # Enforce positional index alignment to prevent silent row misalignment
    if not X.index.equals(metadata_df.index):
        raise ValueError(
            "Index mismatch between feature matrix X and metadata_df. "
            "Ensure both DataFrames share identical row ordering and indices."
        )
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError(
            "Index mismatch between feature matrix X and target Series y. "
            "Ensure X and y share identical row ordering and indices."
        )

    # Guard against accidental ID columns inside X
    id_in_x = sorted(required_meta & set(X.columns))
    if id_in_x:
        raise ValueError(f"Metadata ID columns must not be present in feature matrix X: {id_in_x}")

    raw_splits = (
        splits
        if splits is not None
        else generate_cv_splits(
            y_arr,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    )
    cv_splits = validate_cv_splits(raw_splits, n_samples=n_samples)

    oof_prob = np.full(n_samples, np.nan, dtype=float)
    oof_fold = np.full(n_samples, -1, dtype=int)
    prediction_counts = np.zeros(n_samples, dtype=int)

    fold_metrics: list[dict[str, Any]] = []
    fitted_pipelines: list[Pipeline] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train = X.iloc[train_idx]
        y_train = y_arr[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y_arr[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        proba = fold_pipeline.predict_proba(X_val)
        classes = list(fold_pipeline.classes_)
        if 1 in classes:
            pos_col = classes.index(1)
            val_prob = proba[:, pos_col]
        else:
            val_prob = np.zeros(len(val_idx), dtype=float)

        oof_prob[val_idx] = val_prob
        oof_fold[val_idx] = fold_idx
        prediction_counts[val_idx] += 1

        metrics_dict = compute_binary_metrics(
            y_true=y_val,
            y_prob=val_prob,
            threshold=decision_threshold,
        )
        fold_metrics.append({"fold": fold_idx, **metrics_dict})

        if retain_fold_pipelines:
            fitted_pipelines.append(fold_pipeline)

    if not np.all(prediction_counts == 1):
        raise RuntimeError(
            "OOF coverage invariant violated: every observation must be predicted exactly once."
        )

    oof_pred = (oof_prob >= decision_threshold).astype(int)
    oof_df = pd.DataFrame(
        {
            "subscription_id": metadata_df["subscription_id"].to_numpy(),
            "account_id": metadata_df["account_id"].to_numpy(),
            "target": target,
            "model": model_name,
            "y_true": y_arr,
            "y_probability": oof_prob,
            "y_pred_0_5": oof_pred,
            "fold": oof_fold,
        }
    )

    oof_metrics = compute_binary_metrics(
        y_true=y_arr,
        y_prob=oof_prob,
        threshold=decision_threshold,
    )

    return CVResult(
        scenario=scenario,
        target=target,
        model_name=model_name,
        decision_threshold=decision_threshold,
        threshold_optimized=False,
        class_distribution=compute_class_distribution(y_arr),
        fold_distributions=summarize_fold_class_distributions(y_arr, cv_splits),
        fold_metrics=fold_metrics,
        summary_metrics=summarize_fold_metrics(fold_metrics),
        oof_metrics=oof_metrics,
        oof_predictions=oof_df,
        fitted_fold_pipelines=fitted_pipelines,
    )
