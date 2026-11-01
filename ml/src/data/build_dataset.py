"""
Merged Dataset Builder for SaaS Churn Intelligence EDA.

Aggregates one-to-many payments and invoices into subscription-level summaries,
then joins them with accounts, subscriptions, and plans into a single dataset
at the granularity of One Row Per Account/Subscription.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
DEFAULT_PROCESSED_TABLES_DIR = ML_DIR / "data" / "processed" / "tables"
DEFAULT_OUTPUT_DIR = ML_DIR / "data" / "processed" / "merged"
DEFAULT_REPORTS_DIR = ML_DIR / "reports" / "cleaning"

# Benchmark coverage cutoff date from data dictionary
DATASET_OBSERVATION_CUTOFF = "2025-12-31"


@dataclass
class DatasetBuildSummary:
    unique_accounts: int
    unique_subscriptions: int
    total_merged_rows: int
    total_columns: int
    subscriptions_with_invoices: int
    subscription_invoice_coverage_pct: float
    subscriptions_with_payments: int
    subscription_payment_coverage_pct: float
    is_one_to_one_account_subscription: bool
    churn_rate_pct: float
    output_path: str


class DatasetBuilder:
    """Constructs the unified analysis dataset with proper aggregation and grain checks."""

    def __init__(
        self,
        tables_dir: Path = DEFAULT_PROCESSED_TABLES_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)

    def load_clean_tables(self) -> Dict[str, pd.DataFrame]:
        tables = {}
        for name in ["plans", "accounts", "subscriptions", "invoices", "payments"]:
            path = self.tables_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Cleaned table not found: {path}. Run preprocess.py first.")
            tables[name] = pd.read_csv(path)
        return tables

    def aggregate_payments_to_invoice(self, payments_df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Aggregate payments to the invoice grain (one row per invoice)."""
        logger.info(f"Aggregating {len(payments_df)} payment records to invoice level...")

        pay = payments_df.copy()
        pay["payment_status"] = pay["payment_status"].astype(str).str.strip().str.lower()
        pay["amount"] = pd.to_numeric(pay["amount"], errors="coerce").fillna(0.0)

        # Calculate metrics per invoice
        grouped = pay.groupby("invoice_id")
        inv_agg = pd.DataFrame({
            "total_payment_attempts": grouped["payment_id"].count(),
            "succeeded_payments_count": grouped["payment_status"].apply(lambda s: (s == "succeeded").sum()),
            "failed_payments_count": grouped["payment_status"].apply(lambda s: (s == "failed").sum()),
            "total_paid_amount": grouped.apply(
                lambda g: g.loc[g["payment_status"] == "succeeded", "amount"].sum(),
                include_groups=False,
            ),
            "first_payment_attempt_date": grouped["attempted_at"].min(),
            "last_payment_attempt_date": grouped["attempted_at"].max(),
        }).reset_index()

        logger.info(f"Aggregated payments into {len(inv_agg)} distinct invoice summaries.")
        return inv_agg

    def aggregate_invoices_to_subscription(
        self, invoices_df: pd.DataFrame, invoice_payments_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Step 2: Join invoices with payment aggregates and aggregate to subscription grain."""
        logger.info(f"Joining {len(invoices_df)} invoices with payment aggregates and rolling up to subscription...")

        inv = invoices_df.copy()
        inv["invoice_status"] = inv["invoice_status"].astype(str).str.strip().str.lower()
        inv["amount_due"] = pd.to_numeric(inv["amount_due"], errors="coerce").fillna(0.0)

        # Merge invoice with payment aggregates
        inv_merged = inv.merge(invoice_payments_df, on="invoice_id", how="left")
        inv_merged["total_payment_attempts"] = inv_merged["total_payment_attempts"].fillna(0).astype(int)
        inv_merged["succeeded_payments_count"] = inv_merged["succeeded_payments_count"].fillna(0).astype(int)
        inv_merged["failed_payments_count"] = inv_merged["failed_payments_count"].fillna(0).astype(int)
        inv_merged["total_paid_amount"] = inv_merged["total_paid_amount"].fillna(0.0)

        # Group by subscription_id
        grouped = inv_merged.groupby("subscription_id")
        sub_agg = pd.DataFrame({
            "total_invoices": grouped["invoice_id"].count(),
            "paid_invoices_count": grouped["invoice_status"].apply(lambda s: (s == "paid").sum()),
            "open_invoices_count": grouped["invoice_status"].apply(lambda s: (s == "open").sum()),
            "void_invoices_count": grouped["invoice_status"].apply(lambda s: (s == "void").sum()),
            "total_amount_billed": grouped["amount_due"].sum(),
            "total_amount_paid": grouped["total_paid_amount"].sum(),
            "total_payment_attempts": grouped["total_payment_attempts"].sum(),
            "total_payment_failures": grouped["failed_payments_count"].sum(),
            "total_payment_successes": grouped["succeeded_payments_count"].sum(),
            "first_invoice_date": grouped["invoice_date"].min(),
            "last_invoice_date": grouped["invoice_date"].max(),
        }).reset_index()

        # Derived billing features
        sub_agg["payment_failure_rate"] = np.where(
            sub_agg["total_payment_attempts"] > 0,
            sub_agg["total_payment_failures"] / sub_agg["total_payment_attempts"],
            0.0,
        )
        sub_agg["has_failed_payment"] = sub_agg["total_payment_failures"] > 0

        logger.info(f"Aggregated invoices and payments into {len(sub_agg)} distinct subscription billing records.")
        return sub_agg

    def build_merged_dataset(self) -> Tuple[pd.DataFrame, DatasetBuildSummary]:
        """Step 3: Merge subscriptions, accounts, plans, and subscription-level billing summaries."""
        tables = self.load_clean_tables()
        acc = tables["accounts"].copy()
        sub = tables["subscriptions"].copy()
        plans = tables["plans"].copy()
        inv = tables["invoices"].copy()
        pay = tables["payments"].copy()

        # 1. Verify 1:1 relationship between accounts and subscriptions
        n_unique_acc = acc["account_id"].nunique()
        n_unique_sub = sub["subscription_id"].nunique()
        n_sub_acc = sub["account_id"].nunique()
        is_one_to_one = (n_unique_acc == len(acc)) and (n_unique_sub == len(sub)) and (n_sub_acc == n_unique_acc)

        if not is_one_to_one:
            logger.warning("Accounts and Subscriptions do NOT have a strictly 1:1 relationship!")
        else:
            logger.info("Verified: Accounts and Subscriptions form a perfect 1:1 mapping (2,000 accounts = 2,000 subscriptions).")

        # 2. Perform two-stage aggregation of payments and invoices
        inv_payments = self.aggregate_payments_to_invoice(pay)
        sub_billing = self.aggregate_invoices_to_subscription(inv, inv_payments)

        # 3. Base Merge: Subscriptions + Accounts (1:1)
        merged = sub.merge(acc, on="account_id", how="inner", suffixes=("", "_account"))

        # 4. Join with Plans
        merged = merged.merge(plans, on="plan_id", how="left")

        # 5. Left Join with Subscription Billing Aggregates
        merged = merged.merge(sub_billing, on="subscription_id", how="left")

        # 6. Feature Engineering for EDA & Analytics
        # Indicator flags
        merged["has_billing_data"] = merged["total_invoices"].notna()
        merged["has_payment_data"] = merged["total_payment_attempts"].notna() & (merged["total_payment_attempts"] > 0)

        # Fill billing metrics for accounts without billing data
        billing_num_cols = [
            "total_invoices",
            "paid_invoices_count",
            "open_invoices_count",
            "void_invoices_count",
            "total_amount_billed",
            "total_amount_paid",
            "total_payment_attempts",
            "total_payment_failures",
            "total_payment_successes",
            "payment_failure_rate",
        ]
        for col in billing_num_cols:
            merged[col] = merged[col].fillna(0)

        merged["has_failed_payment"] = merged["has_failed_payment"].fillna(False)

        # Churn target label
        merged["churn"] = (merged["status"].astype(str).str.lower() == "cancelled").astype(int)

        # Tenure calculation
        started_dt = pd.to_datetime(merged["started_at"], errors="coerce")
        ended_dt = pd.to_datetime(merged["ended_at"], errors="coerce")
        cutoff_dt = pd.to_datetime(DATASET_OBSERVATION_CUTOFF)

        # If cancelled, duration = ended_at - started_at; if active, duration = cutoff_date - started_at
        end_ref = ended_dt.combine_first(pd.Series(cutoff_dt, index=merged.index))
        merged["tenure_days"] = (end_ref - started_dt).dt.days

        # Extra seats beyond plan included
        merged["seats"] = pd.to_numeric(merged["seats"], errors="coerce")
        merged["included_seats"] = pd.to_numeric(merged["included_seats"], errors="coerce")
        merged["additional_seats"] = (merged["seats"] - merged["included_seats"]).clip(lower=0)

        # MRR numeric
        merged["mrr"] = pd.to_numeric(merged["mrr"], errors="coerce")
        merged["monthly_price"] = pd.to_numeric(merged["monthly_price"], errors="coerce")

        # Verify final row count preservation
        final_rows = len(merged)
        if final_rows != n_unique_sub:
            raise ValueError(f"Row count mismatch! Expected {n_unique_sub} rows, but got {final_rows}.")

        subs_with_inv = int(merged["has_billing_data"].sum())
        subs_with_pay = int(merged["has_payment_data"].sum())
        churn_rate = float(merged["churn"].mean() * 100)

        summary = DatasetBuildSummary(
            unique_accounts=n_unique_acc,
            unique_subscriptions=n_unique_sub,
            total_merged_rows=final_rows,
            total_columns=len(merged.columns),
            subscriptions_with_invoices=subs_with_inv,
            subscription_invoice_coverage_pct=round((subs_with_inv / final_rows) * 100, 2),
            subscriptions_with_payments=subs_with_pay,
            subscription_payment_coverage_pct=round((subs_with_pay / final_rows) * 100, 2),
            is_one_to_one_account_subscription=is_one_to_one,
            churn_rate_pct=round(churn_rate, 2),
            output_path=str(self.output_dir / "account_level_eda_dataset.csv"),
        )

        return merged, summary

    def run(self) -> DatasetBuildSummary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        merged_df, summary = self.build_merged_dataset()

        # Save merged dataset
        output_path = Path(summary.output_path)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved merged dataset: {output_path} ({summary.total_merged_rows} rows, {summary.total_columns} columns)")

        # Save build summary
        json_report = self.reports_dir / "dataset_build_summary.json"
        with open(json_report, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        logger.info(f"Saved dataset build summary JSON: {json_report}")

        return summary


def main() -> None:
    builder = DatasetBuilder()
    summary = builder.run()
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print(f"Total Rows (Grain: Account/Subscription): {summary.total_merged_rows:,}")
    print(f"Total Columns: {summary.total_columns}")
    print(f"Unique Accounts: {summary.unique_accounts:,}")
    print(f"Unique Subscriptions: {summary.unique_subscriptions:,}")
    print(f"Invoice Coverage: {summary.subscriptions_with_invoices} ({summary.subscription_invoice_coverage_pct}%)")
    print(f"Payment Coverage: {summary.subscriptions_with_payments} ({summary.subscription_payment_coverage_pct}%)")
    print(f"Churn Rate: {summary.churn_rate_pct}%")
    print(f"Output File: {summary.output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
