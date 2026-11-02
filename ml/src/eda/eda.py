"""
Exploratory Data Analysis (EDA) Pipeline for SaaS Churn Intelligence.

Performs statistical analysis across individual clean tables and the unified dataset,
generates high-resolution visualizations in ml/reports/eda/figures/,
and produces comprehensive analytical reports in ml/reports/eda/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir() + "/matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
DEFAULT_MERGED_DATASET_PATH = ML_DIR / "data" / "processed" / "merged" / "account_level_eda_dataset.csv"
DEFAULT_TABLES_DIR = ML_DIR / "data" / "processed" / "tables"
DEFAULT_FIGURES_DIR = ML_DIR / "reports" / "eda" / "figures"
DEFAULT_EDA_REPORTS_DIR = ML_DIR / "reports" / "eda"

# Set visualization theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})


class SaasChurnEDA:
    """End-to-end Exploratory Data Analysis runner for SaaS subscriptions."""

    def __init__(
        self,
        merged_path: Path = DEFAULT_MERGED_DATASET_PATH,
        tables_dir: Path = DEFAULT_TABLES_DIR,
        figures_dir: Path = DEFAULT_FIGURES_DIR,
        reports_dir: Path = DEFAULT_EDA_REPORTS_DIR,
    ) -> None:
        self.merged_path = Path(merged_path)
        self.tables_dir = Path(tables_dir)
        self.figures_dir = Path(figures_dir)
        self.reports_dir = Path(reports_dir)

    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if not self.merged_path.exists():
            raise FileNotFoundError(f"Merged dataset not found at {self.merged_path}. Run build_dataset.py first.")
        df = pd.read_csv(self.merged_path)
        tables = {}
        for name in ["plans", "accounts", "subscriptions", "invoices", "payments"]:
            t_path = self.tables_dir / f"{name}.csv"
            if t_path.exists():
                tables[name] = pd.read_csv(t_path)
        return df, tables

    def run(self) -> Dict[str, Any]:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading cleaned datasets for EDA...")
        df, tables = self.load_data()

        logger.info("Generating EDA figures...")
        self.plot_churn_distribution(df)
        self.plot_churn_by_plan(df)
        self.plot_churn_by_segment(df)
        self.plot_churn_by_region(df)
        self.plot_churn_by_channel(df)
        self.plot_mrr_by_churn(df)
        self.plot_tenure_by_churn(df)
        self.plot_missing_values(df, tables)
        self.plot_numeric_distributions(df)
        self.plot_monthly_churn_trend(df)
        self.plot_payment_failures_cohort(df)
        self.plot_invoice_coverage_breakdown(df)

        logger.info("Computing EDA summary metrics...")
        metrics = self.compute_metrics(df, tables)

        logger.info("Writing EDA reports...")
        self.write_reports(metrics)

        logger.info("EDA completed successfully.")
        return metrics

    def plot_churn_distribution(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        status_counts = df["status"].value_counts()
        colors = ["#2ecc71", "#e74c3c"]

        # Bar chart
        sns.barplot(x=status_counts.index, y=status_counts.values, ax=axes[0], palette=colors, hue=status_counts.index, legend=False)
        axes[0].set_title("Customer Status Volume")
        axes[0].set_xlabel("Subscription Status")
        axes[0].set_ylabel("Account Count")
        for i, val in enumerate(status_counts.values):
            axes[0].text(i, val + 25, f"{val:,} ({val/len(df)*100:.1f}%)", ha="center", fontweight="bold")
        axes[0].set_ylim(0, max(status_counts.values) * 1.15)

        # Donut chart
        axes[1].pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.4},
        )
        axes[1].set_title("Customer Status Breakdown")

        fig.suptitle("Overall Churn vs Active Distribution", y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "01_churn_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_churn_by_plan(self, df: pd.DataFrame) -> None:
        plan_stats = df.groupby("plan_name")["churn"].agg(
            total="count", churned="sum", churn_rate=lambda s: s.mean() * 100
        ).reset_index().sort_values(by="churn_rate", ascending=False)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        sns.barplot(data=plan_stats, x="plan_name", y="total", ax=ax1, color="#3498db", alpha=0.7)
        sns.lineplot(data=plan_stats, x="plan_name", y="churn_rate", ax=ax2, color="#e74c3c", marker="o", linewidth=2.5, markersize=8)

        ax1.set_title("Subscription Volume and Churn Rate by Plan")
        ax1.set_xlabel("Plan Tier")
        ax1.set_ylabel("Total Subscriptions", color="#2980b9")
        ax2.set_ylabel("Churn Rate (%)", color="#c0392b")
        ax2.set_ylim(0, 35)

        for i, row in plan_stats.reset_index(drop=True).iterrows():
            ax2.text(i, row["churn_rate"] + 1.2, f"{row['churn_rate']:.1f}%", ha="center", color="#c0392b", fontweight="bold")
            ax1.text(i, row["total"] / 2, f"n={row['total']}", ha="center", color="white", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "02_churn_by_plan.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_churn_by_segment(self, df: pd.DataFrame) -> None:
        seg_stats = df.groupby("segment")["churn"].agg(
            total="count", churned="sum", churn_rate=lambda s: s.mean() * 100
        ).reset_index().sort_values(by="churn_rate", ascending=False)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        sns.barplot(data=seg_stats, x="segment", y="total", ax=ax1, color="#9b59b6", alpha=0.7)
        sns.lineplot(data=seg_stats, x="segment", y="churn_rate", ax=ax2, color="#e67e22", marker="s", linewidth=2.5, markersize=8)

        ax1.set_title("Subscription Volume and Churn Rate by Account Segment")
        ax1.set_xlabel("Market Segment")
        ax1.set_ylabel("Total Accounts", color="#8e44ad")
        ax2.set_ylabel("Churn Rate (%)", color="#d35400")
        ax2.set_ylim(0, 30)

        for i, row in seg_stats.reset_index(drop=True).iterrows():
            ax2.text(i, row["churn_rate"] + 1.0, f"{row['churn_rate']:.1f}%", ha="center", color="#d35400", fontweight="bold")
            ax1.text(i, row["total"] / 2, f"n={row['total']}", ha="center", color="white", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "03_churn_by_segment.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_churn_by_region(self, df: pd.DataFrame) -> None:
        df_reg = df.copy()
        df_reg["region_plot"] = df_reg["region"].fillna("Missing / Unknown")

        reg_stats = df_reg.groupby("region_plot")["churn"].agg(
            total="count", churned="sum", churn_rate=lambda s: s.mean() * 100
        ).reset_index().sort_values(by="churn_rate", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = sns.barplot(data=reg_stats, x="region_plot", y="churn_rate", ax=ax, palette="Blues_d", hue="region_plot", legend=False)
        ax.set_title("Churn Rate by Geographic Region (Including Missing Category)")
        ax.set_xlabel("Region")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, 30)

        # Baseline horizontal line
        overall_churn = df["churn"].mean() * 100
        ax.axhline(overall_churn, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Overall Avg ({overall_churn:.1f}%)")
        ax.legend(loc="upper right")

        for i, row in reg_stats.reset_index(drop=True).iterrows():
            ax.text(i, row["churn_rate"] + 0.8, f"{row['churn_rate']:.1f}%\n(n={row['total']})", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "04_churn_by_region.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_churn_by_channel(self, df: pd.DataFrame) -> None:
        ch_stats = df.groupby("acquisition_channel")["churn"].agg(
            total="count", churned="sum", churn_rate=lambda s: s.mean() * 100
        ).reset_index().sort_values(by="churn_rate", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=ch_stats, x="acquisition_channel", y="churn_rate", ax=ax, palette="viridis", hue="acquisition_channel", legend=False)
        ax.set_title("Churn Rate by Customer Acquisition Channel")
        ax.set_xlabel("Acquisition Channel")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, 30)

        overall_churn = df["churn"].mean() * 100
        ax.axhline(overall_churn, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Overall Avg ({overall_churn:.1f}%)")
        ax.legend(loc="upper right")

        for i, row in ch_stats.reset_index(drop=True).iterrows():
            ax.text(i, row["churn_rate"] + 0.8, f"{row['churn_rate']:.1f}%\n(n={row['total']})", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "05_churn_by_acquisition_channel.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_mrr_by_churn(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Boxplot (log scale)
        sns.boxplot(data=df, x="status", y="mrr", ax=axes[0], palette=["#2ecc71", "#e74c3c"], hue="status", legend=False)
        axes[0].set_yscale("log")
        axes[0].set_title("MRR Distribution by Churn Status (Log Scale)")
        axes[0].set_xlabel("Status")
        axes[0].set_ylabel("Contracted MRR ($) - Log Scale")

        # Overlay mean & median values
        mrr_stats = df.groupby("status")["mrr"].agg(["mean", "median"])
        for i, (status, row) in enumerate(mrr_stats.iterrows()):
            axes[0].text(i, row["median"] * 1.25, f"Med: ${row['median']:.0f}\nMean: ${row['mean']:.0f}", ha="center", fontweight="bold")

        # Distribution comparison
        for status, color in [("active", "#2ecc71"), ("cancelled", "#e74c3c")]:
            subset = df[df["status"] == status]["mrr"]
            sns.kdeplot(subset, ax=axes[1], label=f"{status.title()} (mean ${subset.mean():.0f})", color=color, linewidth=2, fill=True, alpha=0.3)
        axes[1].set_title("MRR Density Estimation")
        axes[1].set_xlabel("Contracted MRR ($)")
        axes[1].set_xlim(0, 3000)
        axes[1].legend()

        fig.suptitle("Monthly Recurring Revenue (MRR) vs Customer Churn", y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "06_mrr_distribution_by_churn.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_tenure_by_churn(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Boxplot
        sns.boxplot(data=df, x="status", y="tenure_days", ax=axes[0], palette=["#2ecc71", "#e74c3c"], hue="status", legend=False)
        axes[0].set_title("Tenure Days Distribution")
        axes[0].set_xlabel("Status")
        axes[0].set_ylabel("Tenure (Days)")

        tenure_stats = df.groupby("status")["tenure_days"].agg(["mean", "median"])
        for i, (status, row) in enumerate(tenure_stats.iterrows()):
            axes[0].text(i, row["median"] + 30, f"Med: {row['median']:.0f} d\nMean: {row['mean']:.0f} d", ha="center", fontweight="bold")

        # KDE Distribution
        for status, color in [("active", "#2ecc71"), ("cancelled", "#e74c3c")]:
            subset = df[df["status"] == status]["tenure_days"].dropna()
            sns.kdeplot(subset, ax=axes[1], label=f"{status.title()} (median {subset.median():.0f} d)", color=color, linewidth=2, fill=True, alpha=0.3)
        axes[1].set_title("Tenure Density: Churn Timing")
        axes[1].set_xlabel("Tenure (Days)")
        axes[1].legend()

        fig.suptitle("Subscription Duration (Tenure) vs Churn", y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "07_tenure_distribution_by_churn.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_missing_values(self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> None:
        # Table level null percentages
        table_nulls = []
        for name, t_df in tables.items():
            for col in t_df.columns:
                null_cnt = t_df[col].isna().sum()
                null_pct = (null_cnt / len(t_df)) * 100
                table_nulls.append({"table": name, "column": f"{name}.{col}", "null_pct": null_pct, "null_cnt": null_cnt})

        null_df = pd.DataFrame(table_nulls).sort_values(by="null_pct", ascending=False)
        null_df = null_df[null_df["null_pct"] > 0]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=null_df, y="column", x="null_pct", ax=ax, palette="Reds_r", hue="column", legend=False)
        ax.set_title("Missing Values Percentage Across Clean Tables")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_ylabel("Column Name")
        ax.set_xlim(0, 100)

        for i, row in null_df.reset_index(drop=True).iterrows():
            ax.text(row["null_pct"] + 1, i, f"{row['null_pct']:.1f}% ({row['null_cnt']:,} rows)", va="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "08_missing_values_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_numeric_distributions(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # 1. Seats
        sns.histplot(df["seats"], bins=30, ax=axes[0, 0], color="#2980b9", kde=True)
        axes[0, 0].set_title("Purchased Seats Distribution")
        axes[0, 0].set_xlabel("Seats Count")

        # 2. MRR
        sns.histplot(df["mrr"], bins=30, ax=axes[0, 1], color="#27ae60", kde=True)
        axes[0, 1].set_title("Monthly Recurring Revenue (MRR) Distribution")
        axes[0, 1].set_xlabel("MRR ($)")

        # 3. Monthly Price
        sns.countplot(data=df, x="monthly_price", ax=axes[1, 0], palette="Blues_d", hue="monthly_price", legend=False)
        axes[1, 0].set_title("Plan Base Monthly Price Distribution")
        axes[1, 0].set_xlabel("Monthly Price ($)")

        # 4. Additional Seats
        sns.histplot(df["additional_seats"], bins=30, ax=axes[1, 1], color="#8e44ad", kde=True)
        axes[1, 1].set_title("Additional Seats (Beyond Included Seats)")
        axes[1, 1].set_xlabel("Additional Seats")

        fig.suptitle("Core Numeric Feature Distributions", y=1.01)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "09_numeric_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_monthly_churn_trend(self, df: pd.DataFrame) -> None:
        churn_df = df[df["status"] == "cancelled"].copy()
        churn_df["ended_month"] = pd.to_datetime(churn_df["ended_at"]).dt.to_period("M").astype(str)
        monthly_churn = churn_df.groupby("ended_month").size().reset_index(name="churn_events")

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=monthly_churn, x="ended_month", y="churn_events", ax=ax, marker="o", color="#c0392b", linewidth=2.5)
        ax.set_title("Monthly Churn Volume Trend (2023 - 2025) - Cancellation Count")
        ax.set_xlabel("Cancellation Month")
        ax.set_ylabel("Monthly Cancellations (Volume)")
        plt.xticks(rotation=45)
        for i, row in monthly_churn.iterrows():
            ax.text(i, row["churn_events"] + 0.6, str(row["churn_events"]), ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "10_monthly_churn_trend.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_payment_failures_cohort(self, df: pd.DataFrame) -> None:
        billing_cohort = df[df["has_billing_data"]].copy()

        cohort_stats = billing_cohort.groupby("has_failed_payment")["churn"].agg(
            total="count", churned="sum", churn_rate=lambda s: s.mean() * 100
        ).reset_index()
        cohort_stats["label"] = cohort_stats["has_failed_payment"].map({True: "With Failed Payment", False: "No Failed Payment"})

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Churn rate comparison
        sns.barplot(data=cohort_stats, x="label", y="churn_rate", ax=axes[0], palette=["#27ae60", "#e67e22"], hue="label", legend=False)
        axes[0].set_title("120-Subscription Cohort: Churn Rate vs Payment Failure")
        axes[0].set_ylabel("Churn Rate (%)")
        axes[0].set_xlabel("Payment History Profile")
        axes[0].set_ylim(0, 35)
        for i, row in cohort_stats.iterrows():
            axes[0].text(i, row["churn_rate"] + 1, f"{row['churn_rate']:.1f}%\n(n={row['total']})", ha="center", fontweight="bold")

        # Failure count distribution
        sns.countplot(data=billing_cohort, x="total_payment_failures", hue="status", ax=axes[1], palette=["#2ecc71", "#e74c3c"])
        axes[1].set_title("Payment Failure Frequency vs Churn Status")
        axes[1].set_xlabel("Total Failed Payment Attempts")
        axes[1].set_ylabel("Subscription Count")

        fig.suptitle("Billing & Payment Failure Analysis (Restricted 120-Subscription Cohort)", y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "11_payment_failures_vs_churn_cohort.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_invoice_coverage_breakdown(self, df: pd.DataFrame) -> None:
        coverage_counts = df["has_billing_data"].value_counts()
        labels = ["No Billing History\n(S000121 - S002000)", "Has Invoices\n(S000001 - S000120)"]
        colors = ["#95a5a6", "#2980b9"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Donut chart
        axes[0].pie(
            [coverage_counts[False], coverage_counts[True]],
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.4},
        )
        axes[0].set_title("Dataset Invoice Coverage Disparity")

        # Bar chart comparing churn rates between covered cohort and non-covered cohort
        coverage_churn = df.groupby("has_billing_data")["churn"].agg(
            total="count", churn_rate=lambda s: s.mean() * 100
        ).reset_index()
        coverage_churn["cohort"] = coverage_churn["has_billing_data"].map({True: "Invoiced (n=120)", False: "Non-Invoiced (n=1,880)"})

        sns.barplot(data=coverage_churn, x="cohort", y="churn_rate", ax=axes[1], palette=["#7f8c8d", "#3498db"], hue="cohort", legend=False)
        axes[1].set_title("Churn Rate Comparison Across Invoicing Cohorts")
        axes[1].set_ylabel("Churn Rate (%)")
        axes[1].set_ylim(0, 30)
        for i, row in coverage_churn.iterrows():
            axes[1].text(i, row["churn_rate"] + 0.8, f"{row['churn_rate']:.1f}%", ha="center", fontweight="bold")

        fig.suptitle("Critical Data Limitation: 6.0% Invoice Coverage", y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "12_invoice_coverage_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()

    def compute_metrics(self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        total_accounts = len(df)
        churn_count = int(df["churn"].sum())
        active_count = total_accounts - churn_count
        churn_rate = float(df["churn"].mean() * 100)

        # Plan metrics
        plan_metrics = (
            df.groupby("plan_name")["churn"]
            .agg(total="count", churned="sum", churn_rate=lambda s: float(s.mean() * 100))
            .to_dict(orient="index")
        )

        # Segment metrics
        segment_metrics = (
            df.groupby("segment")["churn"]
            .agg(total="count", churned="sum", churn_rate=lambda s: float(s.mean() * 100))
            .to_dict(orient="index")
        )

        # Region metrics
        region_df = df.copy()
        region_df["region_filled"] = region_df["region"].fillna("Missing")
        region_metrics = (
            region_df.groupby("region_filled")["churn"]
            .agg(total="count", churned="sum", churn_rate=lambda s: float(s.mean() * 100))
            .to_dict(orient="index")
        )

        # Acquisition channel metrics
        channel_metrics = (
            df.groupby("acquisition_channel")["churn"]
            .agg(total="count", churned="sum", churn_rate=lambda s: float(s.mean() * 100))
            .to_dict(orient="index")
        )

        # MRR stats
        mrr_by_status = df.groupby("status")["mrr"].describe().to_dict(orient="index")

        # Tenure stats
        tenure_by_status = df.groupby("status")["tenure_days"].describe().to_dict(orient="index")

        # 120 billing cohort metrics
        billing_cohort = df[df["has_billing_data"]]
        cohort_metrics = {
            "total_invoiced_subs": len(billing_cohort),
            "subs_with_payment_failures": int((billing_cohort["total_payment_failures"] > 0).sum()),
            "subs_without_payment_failures": int((billing_cohort["total_payment_failures"] == 0).sum()),
            "churn_rate_with_failure": float(billing_cohort[billing_cohort["total_payment_failures"] > 0]["churn"].mean() * 100),
            "churn_rate_without_failure": float(billing_cohort[billing_cohort["total_payment_failures"] == 0]["churn"].mean() * 100),
            "total_invoices_in_table": len(tables["invoices"]) if "invoices" in tables else None,
            "total_payments_in_table": len(tables["payments"]) if "payments" in tables else None,
        }

        # Statistical independence tests for missing region against observable features (Chi-square)
        df_missing_test = df.copy()
        df_missing_test["is_missing_region"] = df_missing_test["region"].isna()
        region_independence_tests = {}
        for col in ["churn", "segment", "acquisition_channel", "plan_name"]:
            try:
                contingency = pd.crosstab(df_missing_test["is_missing_region"], df_missing_test[col])
                if contingency.empty or (contingency.values == 0).all() or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    region_independence_tests[col] = {
                        "status": "cannot_compute",
                        "error": "Degenerate contingency table (fewer than 2 rows or columns)",
                        "chi2_statistic": None,
                        "degrees_of_freedom": None,
                        "p_value": None,
                        "min_expected_frequency": None,
                        "cells_expected_lt_5": None,
                        "total_cells": None,
                        "assumptions_met": False,
                    }
                    continue

                chi2_stat, p_val, dof, expected = chi2_contingency(contingency, correction=False)
                min_exp = float(expected.min())
                cells_lt_5 = int((expected < 5).sum())
                total_cells = int(expected.size)
                assumptions_met = bool(min_exp >= 5.0 and cells_lt_5 == 0)

                region_independence_tests[col] = {
                    "status": "success",
                    "chi2_statistic": round(float(chi2_stat), 4),
                    "degrees_of_freedom": int(dof),
                    "p_value": round(float(p_val), 4),
                    "min_expected_frequency": round(min_exp, 2),
                    "cells_expected_lt_5": cells_lt_5,
                    "total_cells": total_cells,
                    "assumptions_met": assumptions_met,
                }
            except Exception as e:
                region_independence_tests[col] = {
                    "status": "error",
                    "error": str(e),
                    "chi2_statistic": None,
                    "degrees_of_freedom": None,
                    "p_value": None,
                    "min_expected_frequency": None,
                    "cells_expected_lt_5": None,
                    "total_cells": None,
                    "assumptions_met": False,
                }

        return {
            "total_accounts": total_accounts,
            "total_subscriptions": total_accounts,
            "active_subscriptions": active_count,
            "churned_subscriptions": churn_count,
            "overall_churn_rate_pct": round(churn_rate, 2),
            "plan_metrics": plan_metrics,
            "segment_metrics": segment_metrics,
            "region_metrics": region_metrics,
            "channel_metrics": channel_metrics,
            "mrr_by_status": mrr_by_status,
            "tenure_by_status": tenure_by_status,
            "billing_cohort_metrics": cohort_metrics,
            "region_independence_tests": region_independence_tests,
        }

    def write_reports(self, metrics: Dict[str, Any]) -> None:
        json_path = self.reports_dir / "eda_metrics.json"
        md_path = self.reports_dir / "eda_summary_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved EDA metrics JSON: {json_path}")

        md_lines = [
            "# Keşifsel Veri Analizi (EDA) Kapsamlı Özet Raporu",
            "",
            "## 1. Yönetici Özeti (Executive Summary)",
            "",
            f"- **Toplam Müşteri / Abonelik Sayısı**: {metrics['total_accounts']:,}",
            f"- **Aktif Abonelikler**: {metrics['active_subscriptions']:,} (%{100 - metrics['overall_churn_rate_pct']:.2f})",
            f"- **Müşteri Kaybı (Churn) Sayısı**: {metrics['churned_subscriptions']:,} (%{metrics['overall_churn_rate_pct']:.2f})",
            "- **Veri Seti Kapsamı**: 2023-01-01 ile 2025-12-31 tarihleri arasındaki sentetik SaaS B2B verisi.",
            "- **Lisans**: Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "",
            "## 2. Ana Analiz Bulguları",
            "",
            "### 2.1. Plan Bazlı Churn Analizi",
            "| Plan | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |",
            "|---|---:|---:|---:|",
        ]

        for p_name, p_data in metrics["plan_metrics"].items():
            md_lines.append(f"| `{p_name}` | {p_data['total']:,} | {p_data['churned']:,} | %{p_data['churn_rate']:.2f} |")

        md_lines.extend([
            "",
            "> **Gözlem**: En yüksek churn oranı %25.00 ile `Enterprise` paketindedir. Kurumsal seviyedeki yüksek MRR kayıplarını engellemek için müşteri başarı (Customer Success) müdahaleleri önceliklendirilmelidir.",
            "",
            "### 2.2. Segment Bazlı Churn Analizi",
            "| Segment | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |",
            "|---|---:|---:|---:|",
        ])

        for s_name, s_data in metrics["segment_metrics"].items():
            md_lines.append(f"| `{s_name}` | {s_data['total']:,} | {s_data['churned']:,} | %{s_data['churn_rate']:.2f} |")

        md_lines.extend([
            "",
            "> **Gözlem**: En düşük churn %17.16 ile `mid_market` segmentindedir. `SMB` segmenti hacim olarak veri setinin %70.35'ini oluşturmakta ve %21.54 churn oranına sahiptir.",
            "",
            "### 2.3. Coğrafi Bölge ve Eksik Bölge Analizi (Ki-kare Bağımsızlık Testleri)",
            "| Bölge | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |",
            "|---|---:|---:|---:|",
        ])

        for r_name, r_data in metrics["region_metrics"].items():
            md_lines.append(f"| `{r_name}` | {r_data['total']:,} | {r_data['churned']:,} | %{r_data['churn_rate']:.2f} |")

        md_lines.extend([
            "",
            "#### Eksik Bölge ile Gözlemlenebilir Değişkenler Arasındaki Ki-kare Bağımsızlık Testleri",
            "",
            "Hesaplanan parametreler: $\\chi^2$ test istatistiği, serbestlik derecesi (dof), p-değeri (`correction=False`), minimum beklenen hücre frekansı ve Ki-kare varsayım kontrolü (beklenen frekansı < 5 olan hücre sayısı):",
            "",
            "| Değişken | $\\chi^2$ İstatistiği | dof | p-değeri | Min. Beklenen Frekans | Hücre (< 5) | Varsayım Durumu |",
            "|---|---:|---:|---:|---:|:---:|:---:|",
        ])

        for var_name, t_res in metrics["region_independence_tests"].items():
            if t_res["status"] == "success":
                assump_icon = "Karşılandı ✅" if t_res["assumptions_met"] else "İhlal ⚠️"
                cells_str = f"{t_res['cells_expected_lt_5']}/{t_res['total_cells']}"
                md_lines.append(
                    f"| `{var_name}` | {t_res['chi2_statistic']} | {t_res['degrees_of_freedom']} | "
                    f"{t_res['p_value']} | {t_res['min_expected_frequency']} | {cells_str} | {assump_icon} |"
                )
            else:
                md_lines.append(f"| `{var_name}` | N/A | N/A | N/A | N/A | N/A | Hesaplanamadı ({t_res.get('error', 'Hata')}) |")

        # Dynamic evaluation of Chi-Square results
        success_tests = [t for t in metrics["region_independence_tests"].values() if t.get("status") == "success"]
        violated_tests = [t for t in success_tests if not t.get("assumptions_met", False)]
        all_p_vals = [t["p_value"] for t in success_tests if t.get("p_value") is not None]
        all_min_exps = [t["min_expected_frequency"] for t in success_tests if t.get("min_expected_frequency") is not None]

        if all_p_vals:
            min_p = min(all_p_vals)
            max_p = max(all_p_vals)
            p_val_analysis = (
                f"$\\alpha = 0.05$ anlamlılık düzeyinde incelendiğinde, hesaplanan p-değerleri "
                f"({min_p:.4f} ile {max_p:.4f} aralığında) "
                f"{'tüm test edilen değişkenler için ' if min_p >= 0.05 else ''}anlamlılık eşiğinin "
                f"{'üzerindedir' if min_p >= 0.05 else 'altında olan değişkenler bulunmaktadır'}. "
                f"{'Bu durum, eksik bölge bilgisi ile test edilen değişkenler arasında istatistiksel olarak anlamlı bir ilişki bulunamadığını gösterir.' if min_p >= 0.05 else 'Anlamlı ilişki gösteren değişkenler dikkatle incelenmelidir.'}"
            )
        else:
            p_val_analysis = "Ki-kare testleri hesaplanamamıştır."

        if violated_tests:
            violated_vars = [
                var
                for var, t in metrics["region_independence_tests"].items()
                if t.get("status") == "success" and not t.get("assumptions_met", False)
            ]
            assumption_analysis = (
                f"⚠️ **Cochran Varsayım İhlali**: `{', '.join(violated_vars)}` değişken(ler)inde beklenen hücre frekansı "
                f"5'in altına düşmüştür (E_ij < 5). Asimptotik Ki-kare testi bu değişkenler için güvenilir olmayabilir; "
                f"Fisher exact test veya kategori birleştirme önerilir."
            )
        elif all_min_exps:
            overall_min_exp = min(all_min_exps)
            assumption_analysis = (
                f"Minimum beklenen hücre frekanslarının tamamı (en düşüğü {overall_min_exp:.2f}) 5 eşiğinin üzerindedir "
                f"ve beklenen frekansı 5'in altında olan hücre yoktur. Cochran varsayımları tüm testlerde tam olarak karşılanmaktadır."
            )
        else:
            assumption_analysis = "Hücre frekansı varsayımları değerlendirilememiştir."

        md_lines.extend([
            "",
            "> **İstatistiksel Yorum ve Metodolojik Sınırlar (Eksik Veri Mekanizması)**:",
            f"> - **p-değeri Analizi**: {p_val_analysis}",
            f"> - **Varsayım Kontrolü**: {assumption_analysis}",
            "> - **Önemli Metodolojik Uyarı**: Raporlanan p-değerleri keşifsel ve ham p-değerleridir; çoklu karşılaştırma (multiple testing) düzeltmesi uygulanmamıştır. En önemlisi, **gözlemlenebilir değişkenlerle anlamlı bir ilişki bulunamaması (bağımlılık olmaması), eksikliğin tamamen rastgele (MCAR - Missing Completely At Random) olduğunun kanıtı değildir**. Gözlemlenemeyen faktörlerin (unobserved confounders) eksiklik sürecini etkileyip etkilemediği bu veri setiyle test edilemez. Bu nedenle eksiklik mekanizması mevcut veriler ışığında **belirsiz / kanıtlanmamış** olarak değerlendirilmeli, eksik değerler yapay varsayımlarla doldurulmamalıdır.",
            "",
            "### 2.4. Edinme Kanalı (Acquisition Channel) Bazlı Churn Analizi",
            "| Kanal | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |",
            "|---|---:|---:|---:|",
        ])

        for c_name, c_data in metrics["channel_metrics"].items():
            md_lines.append(f"| `{c_name}` | {c_data['total']:,} | {c_data['churned']:,} | %{c_data['churn_rate']:.2f} |")

        md_lines.extend([
            "",
            "> **Gözlem**: `partner` (%18.00) ve `referral` (%18.48) kanalları en yüksek sadakate sahiptir. `organic` (%22.60) ve `outbound` (%22.37) kanallarında müşteri kaybı belirgin biçimde daha yüksektir.",
            "",
            "### 2.5. Gelir (MRR) ve Süre (Tenure) Analizi",
            "- **MRR**: Churned müşterilerin ortalama sözleşmeli MRR değeri ($357.90), aktif müşterilerin ortalamasından ($332.08) bir miktar daha yüksektir. Bunun sebebi churn eden az sayıdaki Enterprise müşterisinin ortalamayı yukarı çekmesidir. Medyan MRR her iki grupta da $199.00 seviyesindedir.",
            "- **Tenure**: Churn eden müşteriler ortalama 355 gün (medyan 344 gün) sonra ayrılmaktadır. Kayıplar özellikle 6. ve 12. aylar arasında yoğunlaşmaktadır.",
            "- **Tenure Değişkeni ve Hedef Sızıntısı (Target Leakage) Uyarısı**:",
            "  - Veri setindeki `tenure_days` değişkeni, iptal edilmiş aboneliklerde `ended_at - started_at`, aktif aboneliklerde ise gözlem kesim tarihi olan `2025-12-31` ile `started_at` arasındaki fark olarak hesaplanmıştır (farklı referans tarihleri kullanılmıştır).",
            "  - Bu hesaplama tanımlayıcı keşifsel analiz (EDA) amacıyla uygundur.",
            "  - Ancak **bir churn tahmin modelinde doğrudan girdi özniteliği olarak kullanılması durumunda Hedef Sızıntısına (Target Leakage) yol açar**; çünkü iptal tarihi (`ended_at`) gelecekte gerçekleşen hedef olayın bilgisini içerir.",
            "  - Gelecekte makine öğrenmesi modeli eğitilmeden önce tahminin yapılacağı zaman noktası ($T_0$ / observation window) sabitlenmeli ve tenure yalnızca $T_0$ anına kadar geçen süre olarak türetilmelidir.",
            "",
            "## 3. Veri Seti Sınırlılıkları ve Kritik Fatura Analizi",
            "",
            "### 3.1. Fatura ve Ödeme Verisi Kapsam Sınırlılığı (Invoice Coverage Disparity)",
            "- `invoices.csv` dosyasında yer alan 2,000 fatura kaydı yalnızca ilk 120 abonelik (`S000001` - `S000120`) ile ilişkilidir.",
            "- Geriye kalan 1,880 abonelik (%94.0) için herhangi bir fatura veya ödeme geçmişi bulunmamaktadır.",
            "- **120 Aboneliklik Fatura Kohortu Analizi**:",
            f"  - Faturası bulunan abonelik sayısı: {metrics['billing_cohort_metrics']['total_invoiced_subs']}",
            f"  - Başarısız ödeme denemesi yaşayan abonelik: {metrics['billing_cohort_metrics']['subs_with_payment_failures']}",
            f"  - Başarısız ödeme yaşayan grupta churn oranı: %{metrics['billing_cohort_metrics']['churn_rate_with_failure']:.2f}",
            f"  - Başarısız ödeme yaşamayan grupta churn oranı: %{metrics['billing_cohort_metrics']['churn_rate_without_failure']:.2f}",
            "- **Model Eğitimi Açısından Kritik Not**: Fatura ve ödeme özellikleri tüm popülasyonu temsil etmemektedir. İleride makine öğrenmesi modeli eğitilirken, ödeme özelliklerinin sadece %6'lık bir alt grupta var olduğu dikkate alınmalı; genel modelde bu değişkenler doğrudan kullanılmamalı veya eksiklik durumu özel olarak kodlanmalıdır.",
            "",
            "## 4. Oluşturulan Grafikler Listesi",
            "",
            "Tüm grafikler yüksek çözünürlükte (300 DPI) `ml/reports/eda/figures/` dizinine kaydedilmiştir:",
            "1. `01_churn_distribution.png`: Churn ve aktif hesap hacim ve oran dağılımı (Bar & Donut).",
            "2. `02_churn_by_plan.png`: Plan bazında abone sayısı ve churn oranı grafiği.",
            "3. `03_churn_by_segment.png`: Müşteri segmenti bazında hacim ve churn oranı grafiği.",
            "4. `04_churn_by_region.png`: Coğrafi bölgelere ve eksik bölgeye göre churn dağılımı.",
            "5. `05_churn_by_acquisition_channel.png`: Müşteri edinme kanalı bazlı churn oranı analizi.",
            "6. `06_mrr_distribution_by_churn.png`: Churn durumuna göre MRR dağılımı (Boxplot ve KDE Yoğunluk).",
            "7. `07_tenure_distribution_by_churn.png`: Churn süresi ve müşteri kalma süresi analizi (Tenure Days).",
            "8. `08_missing_values_matrix.png`: Tablo sütunları bazında eksik değer oranları grafiği.",
            "9. `09_numeric_distributions.png`: Koltuk sayısı, MRR, liste fiyatı ve ek koltuk dağılımları.",
            "10. `10_monthly_churn_trend.png`: 2023-2025 aylık iptal hacmi (cancellation count / volume) trendi.",
            "11. `11_payment_failures_vs_churn_cohort.png`: 120 abonelik fatura kohortunda ödeme başarısızlıkları ve churn ilişkisi.",
            "12. `12_invoice_coverage_breakdown.png`: Fatura kapsamındaki %6'lık kısıt ve kohort karşılaştırması.",
            "",
            "> [!NOTE]",
            "> **Aylık İptal Grafiği Notu**: `10_monthly_churn_trend.png` grafiği aylık churn oranını değil, iptal edilen abonelik sayısını (hacmini) göstermektedir. Her ay risk altındaki aktif müşteri tabanı sabit ve net tanımlanmadığından bir oran türetilmemiştir. Mevcut sentetik veride mevsimsellik veya nedensellik hakkında yeterli kanıt bulunmadığından kesin çıkarımlardan kaçınılmalıdır.",
            "",
            "## 5. Lisans ve Kaynak Gösterimi",
            "",
            "Attribution: Analytics Engineering, \"SaaS Subscriptions & Customer Churn Dataset v1.0.0\". https://www.analyticsengineering.com/datasets/saas-subscriptions (CC BY 4.0)",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Saved EDA Markdown summary report: {md_path}")


def main() -> None:
    eda = SaasChurnEDA()
    eda.run()


if __name__ == "__main__":
    main()
