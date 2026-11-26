"""
Evaluation Visualization Utilities.

Generates publication-ready matplotlib figures for baseline model evaluation and
scenario comparison (`region_included` vs `region_excluded`):
  - Precision-Recall (PR) curves with Average Precision (primary metric)
  - ROC curves with ROC-AUC
  - Side-by-side scenario comparison bar charts
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_pr_curves(
    oof_by_model: Mapping[str, pd.DataFrame],
    target: str,
    output_path: Path,
    scenario: str | None = None,
) -> Path:
    """Plot Precision-Recall curves from OOF predictions for multiple models."""
    if not oof_by_model:
        raise ValueError("oof_by_model must not be empty.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_rate: float | None = None

    for model_name, oof_df in oof_by_model.items():
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_prob = oof_df["y_probability"].to_numpy(dtype=float)
        if baseline_rate is None:
            baseline_rate = float(np.mean(y_true))
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.4f})")

    if baseline_rate is not None:
        ax.axhline(
            baseline_rate,
            linestyle="--",
            color="gray",
            linewidth=1.2,
            label=f"Prevalence ({baseline_rate:.4f})",
        )

    title_suffix = f" [{scenario}]" if scenario else ""
    ax.set_title(f"Precision-Recall Curve (OOF) — {target}{title_suffix}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_curves(
    oof_by_model: Mapping[str, pd.DataFrame],
    target: str,
    output_path: Path,
    scenario: str | None = None,
) -> Path:
    """Plot ROC curves from OOF predictions for multiple models."""
    if not oof_by_model:
        raise ValueError("oof_by_model must not be empty.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, oof_df in oof_by_model.items():
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_prob = oof_df["y_probability"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random (0.5000)")

    title_suffix = f" [{scenario}]" if scenario else ""
    ax.set_title(f"ROC Curve (OOF) — {target}{title_suffix}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_scenario_comparison(
    comparison_df: pd.DataFrame,
    target: str,
    metric: str,
    output_path: Path,
) -> Path:
    """
    Plot side-by-side bars comparing `region_included` vs `region_excluded`
    CV mean scores (with std error bars) for a specific target and metric.
    """
    subset = comparison_df[comparison_df["target"] == target].copy()
    if subset.empty:
        raise ValueError(f"No rows found in comparison_df for target '{target}'.")

    inc_mean_col = f"region_included_{metric}_mean"
    inc_std_col = f"region_included_{metric}_std"
    exc_mean_col = f"region_excluded_{metric}_mean"
    exc_std_col = f"region_excluded_{metric}_std"

    required_cols = {inc_mean_col, inc_std_col, exc_mean_col, exc_std_col, "model"}
    missing = sorted(required_cols - set(subset.columns))
    if missing:
        raise ValueError(f"comparison_df is missing required columns: {missing}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = subset["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        subset[inc_mean_col].to_numpy(dtype=float),
        width,
        yerr=subset[inc_std_col].to_numpy(dtype=float),
        capsize=4,
        label="region_included",
    )
    ax.bar(
        x + width / 2,
        subset[exc_mean_col].to_numpy(dtype=float),
        width,
        yerr=subset[exc_std_col].to_numpy(dtype=float),
        capsize=4,
        label="region_excluded",
    )

    ax.set_title(f"Scenario Comparison ({metric}) — {target}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(f"{metric} (CV Mean ± Std)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
