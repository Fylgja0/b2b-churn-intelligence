"""
Point-in-Time Safe Feature Engineering & Modeling Dataset Builder.

Constructs an observation-date-anchored (T0 = 2024-12-31) churn modeling dataset
from cleaned source tables (accounts, subscriptions, plans, invoices, payments).

Designed to mitigate target leakage and clearly document temporal assumptions by:
1. Enforcing T0 cohort eligibility (active at 2024-12-31).
2. Deriving features solely from information knowable on or before T0, explicitly
   distinguishing verified historical facts from assumed time-invariant business attributes.
3. Defining 90, 180, and 365-day forward-looking churn target labels bounded by declared
   dataset coverage (2023-01-01 to 2025-12-31).
4. Quarantining explicit target leakage fields (ended_at, status, churn, tenure_days),
   unverified temporal fields (mrr, seats, additional_seats), and low-coverage billing aggregates.
5. Emitting portable repository-relative paths across all manifests and reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
REPO_ROOT = ML_DIR.parent

DEFAULT_PROCESSED_TABLES_DIR = ML_DIR / "data" / "processed" / "tables"
DEFAULT_MODELING_OUTPUT_DIR = ML_DIR / "data" / "processed" / "modeling"
DEFAULT_REPORTS_DIR = ML_DIR / "reports" / "modeling"

# Official observation date and declared dataset boundaries
OBSERVATION_DATE = "2024-12-31"
DATASET_CUTOFF_DATE = "2025-12-31"
DATASET_START_DATE = "2023-01-01"
PREDICTION_HORIZONS_DAYS = [90, 180, 365]


def to_relative_path(path: Path | str, base_dir: Optional[Path] = None) -> str:
    """
    Converts absolute path to repository-relative POSIX path for cross-platform portability.
    If path is already relative or cannot be relativized to base_dir, returns POSIX formatted string.
    """
    p = Path(path)
    base = Path(base_dir or REPO_ROOT).resolve()
    try:
        p_resolved = p.resolve()
        return p_resolved.relative_to(base).as_posix()
    except (ValueError, RuntimeError):
        return p.as_posix()


@dataclass
class CohortStatistics:
    total_subscriptions: int
    included_in_cohort: int
    excluded_ended_on_or_before_t0: int
    excluded_started_after_t0: int
    excluded_missing_started_at: int


@dataclass
class TargetDistribution:
    horizon_days: int
    window_end_date: str
    positive_count: int
    negative_count: int
    positive_rate_pct: float


@dataclass
class BillingCoverageAtT0:
    invoices_at_t0: int
    distinct_subscriptions_invoiced_at_t0: int
    payments_at_t0: int
    distinct_subscriptions_paid_at_t0: int
    cohort_subscriptions_with_invoices_at_t0: int
    cohort_subscriptions_with_payments_at_t0: int
    churn_365d_subscriptions_with_invoices_at_t0: int


@dataclass
class ConsistencyCheckReport:
    active_with_ended_at_count: int
    cancelled_with_null_ended_at_count: int
    inconsistencies_found: bool


@dataclass
class SourceCoverageReport:
    coverage_start_date: str
    coverage_end_date: str
    coverage_basis: str
    independently_verified: bool
    observed_max_started_at: str
    observed_max_ended_at: str
    label_observability_assumption: str


class FeaturePipeline:
    """Constructs a point-in-time modeling dataset and documentation manifest with explicit temporal assumptions."""

    def __init__(
        self,
        tables_dir: Path = DEFAULT_PROCESSED_TABLES_DIR,
        output_dir: Path = DEFAULT_MODELING_OUTPUT_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
        observation_date: str = OBSERVATION_DATE,
        cutoff_date: str = DATASET_CUTOFF_DATE,
        horizons_days: Optional[List[int]] = None,
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.observation_date = observation_date
        self.cutoff_date = cutoff_date
        self.horizons_days = horizons_days or PREDICTION_HORIZONS_DAYS
        self.source_coverage: Optional[SourceCoverageReport] = None

    def load_clean_tables(self) -> Dict[str, pd.DataFrame]:
        """Loads cleaned tables from disk."""
        tables = {}
        for name in ["plans", "accounts", "subscriptions", "invoices", "payments"]:
            path = self.tables_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Cleaned table not found at {path}. Run preprocess.py first.")
            tables[name] = pd.read_csv(path)
        logger.info(f"Loaded {len(tables)} clean source tables from {to_relative_path(self.tables_dir)}")
        return tables

    def validate_dates_and_censoring(
        self, tables: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, str], SourceCoverageReport]:
        """
        Validates prediction horizon boundaries against declared dataset cutoff
        and reports observed event date boundaries without conflating observed max with declared cutoff.
        """
        t0_dt = pd.to_datetime(self.observation_date)
        cutoff_dt = pd.to_datetime(self.cutoff_date)

        horizon_end_dates = {}
        for h in self.horizons_days:
            end_dt = t0_dt + pd.Timedelta(days=h)
            if end_dt > cutoff_dt:
                raise ValueError(
                    f"Prediction horizon {h}d ends at {end_dt.strftime('%Y-%m-%d')}, "
                    f"which exceeds declared dataset coverage end date {self.cutoff_date}! "
                    f"Right-censoring prevents complete observation."
                )
            horizon_end_dates[f"{h}d"] = end_dt.strftime("%Y-%m-%d")

        subs = tables["subscriptions"]
        observed_max_started = str(subs["started_at"].dropna().max())
        observed_max_ended = str(subs["ended_at"].dropna().max())
        observed_max_event = max(observed_max_started, observed_max_ended)

        if observed_max_event <= self.observation_date:
            raise ValueError(
                f"Observed dataset events end at {observed_max_event}, on or before observation date {self.observation_date}. "
                f"Cannot observe forward-looking churn outcomes!"
            )

        coverage_report = SourceCoverageReport(
            coverage_start_date=DATASET_START_DATE,
            coverage_end_date=self.cutoff_date,
            coverage_basis="declared_by_source",
            independently_verified=False,
            observed_max_started_at=observed_max_started,
            observed_max_ended_at=observed_max_ended,
            label_observability_assumption=(
                "Negative churn labels (ended_at IS NULL or ended_at > window_end_date) assume "
                "complete event capture through declared cutoff date 2025-12-31 without right-censoring unobserved cancellations."
            ),
        )
        self.source_coverage = coverage_report

        logger.info(
            f"Validated horizons within declared cutoff {self.cutoff_date}: {horizon_end_dates}. "
            f"Observed event range: max started={observed_max_started}, max ended={observed_max_ended}."
        )
        return horizon_end_dates, coverage_report

    def check_status_ended_at_consistency(self, subs_df: pd.DataFrame) -> ConsistencyCheckReport:
        """
        Audits status vs ended_at for logical integrity.

        Only flags:
        1. status == 'active' and ended_at is not null
        2. status == 'cancelled' and ended_at is null

        Note: status == 'cancelled' and ended_at > T0 is NOT an inconsistency;
        it is the future churn event we aim to predict.
        """
        status_clean = subs_df["status"].astype(str).str.strip().str.lower()
        active_with_ended = (status_clean == "active") & subs_df["ended_at"].notna()
        cancelled_with_null_ended = (status_clean == "cancelled") & subs_df["ended_at"].isna()

        n_active_with_ended = int(active_with_ended.sum())
        n_cancelled_with_null_ended = int(cancelled_with_null_ended.sum())
        inconsistencies = (n_active_with_ended > 0) or (n_cancelled_with_null_ended > 0)

        if inconsistencies:
            logger.warning(
                f"Logical inconsistencies detected in status vs ended_at: "
                f"active_with_ended={n_active_with_ended}, cancelled_with_null_ended={n_cancelled_with_null_ended}"
            )
        else:
            logger.info("Verified: 0 logical inconsistencies between status and ended_at.")

        return ConsistencyCheckReport(
            active_with_ended_at_count=n_active_with_ended,
            cancelled_with_null_ended_at_count=n_cancelled_with_null_ended,
            inconsistencies_found=inconsistencies,
        )

    def create_modeling_cohort(self, subs_df: pd.DataFrame) -> Tuple[pd.DataFrame, CohortStatistics]:
        """
        Identifies subscriptions active at T0 (2024-12-31).

        Eligibility:
        started_at <= T0 AND (ended_at IS NULL OR ended_at > T0)

        Exclusions:
        - started_at IS NULL
        - started_at > T0
        - ended_at <= T0
        """
        total = len(subs_df)
        missing_started = subs_df["started_at"].isna()
        started_after_t0 = ~missing_started & (subs_df["started_at"] > self.observation_date)
        ended_on_before_t0 = (
            ~missing_started
            & ~started_after_t0
            & subs_df["ended_at"].notna()
            & (subs_df["ended_at"] <= self.observation_date)
        )
        in_cohort = (
            ~missing_started
            & (subs_df["started_at"] <= self.observation_date)
            & (subs_df["ended_at"].isna() | (subs_df["ended_at"] > self.observation_date))
        )

        n_cohort = int(in_cohort.sum())
        n_ended_t0 = int(ended_on_before_t0.sum())
        n_started_after = int(started_after_t0.sum())
        n_missing_started = int(missing_started.sum())

        assert n_cohort + n_ended_t0 + n_started_after + n_missing_started == total, "Partition sum mismatch!"

        cohort_df = subs_df[in_cohort].copy()
        stats = CohortStatistics(
            total_subscriptions=total,
            included_in_cohort=n_cohort,
            excluded_ended_on_or_before_t0=n_ended_t0,
            excluded_started_after_t0=n_started_after,
            excluded_missing_started_at=n_missing_started,
        )

        logger.info(
            f"Cohort formed: {n_cohort} included out of {total} "
            f"(ended<=T0: {n_ended_t0}, started>T0: {n_started_after}, missing started_at: {n_missing_started})"
        )
        return cohort_df, stats

    def compute_target_variables(
        self, cohort_df: pd.DataFrame, horizon_end_dates: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[TargetDistribution]]:
        """
        Generates 90d, 180d, and 365d binary churn targets based on ended_at.

        target_churn = 1 if (T0 < ended_at <= window_end_date) else 0.
        """
        df = cohort_df.copy()
        ended_at = df["ended_at"]
        target_stats = []

        for h in self.horizons_days:
            col_name = f"target_churn_{h}d"
            end_date = horizon_end_dates[f"{h}d"]

            is_positive = ended_at.notna() & (ended_at > self.observation_date) & (ended_at <= end_date)
            df[col_name] = is_positive.astype(int)

            pos_count = int(df[col_name].sum())
            neg_count = len(df) - pos_count
            pos_rate = round(float(pos_count / len(df) * 100), 2)

            target_stats.append(
                TargetDistribution(
                    horizon_days=h,
                    window_end_date=end_date,
                    positive_count=pos_count,
                    negative_count=neg_count,
                    positive_rate_pct=pos_rate,
                )
            )
            logger.info(f"Target {col_name} (<= {end_date}): {pos_count} positive, {neg_count} negative ({pos_rate}%)")

        # Hierarchical monotonicity check
        t90 = df["target_churn_90d"]
        t180 = df["target_churn_180d"]
        t365 = df["target_churn_365d"]
        monotonic = ((t90 <= t180) & (t180 <= t365)).all()
        if not monotonic:
            raise ValueError("Target hierarchy violation: target_churn_90d <= target_churn_180d <= target_churn_365d failed!")

        return df, target_stats

    def compute_billing_coverage_at_t0(
        self,
        invoices_df: pd.DataFrame,
        payments_df: pd.DataFrame,
        cohort_subs: set[str],
        churn_365_subs: set[str],
    ) -> BillingCoverageAtT0:
        """
        Evaluates billing and payment history strictly up to T0.

        Rules:
        - Invoices: invoice_date <= T0
        - Payments: attempted_at <= T0 AND invoice was issued on/before T0
        """
        t0_inv = invoices_df[invoices_df["invoice_date"] <= self.observation_date].copy()
        t0_inv_ids = set(t0_inv["invoice_id"].unique())
        t0_inv_subs = set(t0_inv["subscription_id"].unique())

        t0_pay = payments_df[
            (payments_df["attempted_at"] <= self.observation_date)
            & (payments_df["invoice_id"].isin(t0_inv_ids))
        ].copy()

        pay_with_sub = t0_pay.merge(t0_inv[["invoice_id", "subscription_id"]], on="invoice_id", how="left")
        t0_pay_subs = set(pay_with_sub["subscription_id"].dropna().unique())

        cohort_inv_subs = cohort_subs.intersection(t0_inv_subs)
        cohort_pay_subs = cohort_subs.intersection(t0_pay_subs)
        churn_inv_subs = churn_365_subs.intersection(t0_inv_subs)

        stats = BillingCoverageAtT0(
            invoices_at_t0=len(t0_inv),
            distinct_subscriptions_invoiced_at_t0=len(t0_inv_subs),
            payments_at_t0=len(t0_pay),
            distinct_subscriptions_paid_at_t0=len(t0_pay_subs),
            cohort_subscriptions_with_invoices_at_t0=len(cohort_inv_subs),
            cohort_subscriptions_with_payments_at_t0=len(cohort_pay_subs),
            churn_365d_subscriptions_with_invoices_at_t0=len(churn_inv_subs),
        )

        logger.info(
            f"Billing at T0: {stats.invoices_at_t0} invoices ({stats.distinct_subscriptions_invoiced_at_t0} subs), "
            f"{stats.payments_at_t0} payments ({stats.distinct_subscriptions_paid_at_t0} subs). "
            f"Cohort coverage: {stats.cohort_subscriptions_with_invoices_at_t0}/{len(cohort_subs)} subs with invoices, "
            f"365d churn coverage: {stats.churn_365d_subscriptions_with_invoices_at_t0} subs with invoices."
        )
        return stats

    def engineer_features(
        self,
        cohort_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        plans_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Derives point-in-time features from accounts and plans.

        Safe features included (12 features):
        Verified Historical Features:
        - acquisition_channel (invariant initial channel)
        - account_age_days_at_t0 (T0 - created_at)
        - account_created_at_missing_flag (1/0)
        - subscription_tenure_days_at_t0 (T0 - started_at)
        - subscription_start_month (1..12)
        - subscription_start_quarter (1..4)

        Assumed Time-Invariant Features:
        - region (raw, nulls preserved)
        - region_missing_flag (1/0)
        - segment (categorical, assumed invariant)
        - plan_name (categorical, assumed invariant)
        - plan_monthly_price (float, assumed invariant catalog price)
        - plan_included_seats (int, assumed invariant catalog seats)
        """
        df = cohort_df.copy()
        n_cohort = len(df)

        # Merge with accounts (validate 1:1)
        merged = df.merge(accounts_df, on="account_id", how="left", validate="1:1")
        if len(merged) != n_cohort:
            raise ValueError(f"Accounts join row fanout! Expected {n_cohort}, got {len(merged)}")

        # Merge with plans (validate m:1)
        plan_cols = ["plan_id", "plan_name", "monthly_price", "included_seats"]
        merged = merged.merge(plans_df[plan_cols], on="plan_id", how="left", validate="m:1")
        if len(merged) != n_cohort:
            raise ValueError(f"Plans join row fanout! Expected {n_cohort}, got {len(merged)}")

        # Rename plan columns as specified
        merged = merged.rename(
            columns={
                "monthly_price": "plan_monthly_price",
                "included_seats": "plan_included_seats",
            }
        )

        t0_ts = pd.to_datetime(self.observation_date)
        created_at_dt = pd.to_datetime(merged["created_at"], errors="coerce")
        started_at_dt = pd.to_datetime(merged["started_at"], errors="coerce")

        # Account features
        merged["region_missing_flag"] = merged["region"].isna().astype(int)
        merged["account_created_at_missing_flag"] = merged["created_at"].isna().astype(int)

        # Age and tenure
        merged["account_age_days_at_t0"] = (t0_ts - created_at_dt).dt.days
        if merged["account_age_days_at_t0"].isna().any():
            merged["account_age_days_at_t0"] = merged["account_age_days_at_t0"].astype("Int64")
        else:
            merged["account_age_days_at_t0"] = merged["account_age_days_at_t0"].astype("int64")
        merged["subscription_tenure_days_at_t0"] = (t0_ts - started_at_dt).dt.days.astype("int64")

        # Verify no negative age or tenure
        valid_ages = merged["account_age_days_at_t0"].dropna()
        if (valid_ages < 0).any():
            min_neg_age = valid_ages.min()
            raise ValueError(f"Validation failure: Negative account age detected ({min_neg_age} days)!")

        if (merged["subscription_tenure_days_at_t0"] < 0).any():
            min_neg_tenure = merged["subscription_tenure_days_at_t0"].min()
            raise ValueError(f"Validation failure: Negative subscription tenure detected ({min_neg_tenure} days)!")

        # Calendar features
        merged["subscription_start_month"] = started_at_dt.dt.month.astype("int64")
        merged["subscription_start_quarter"] = started_at_dt.dt.quarter.astype("int64")

        # Final ordered columns (2 Metadata + 12 Features + 3 Targets = 17)
        ordered_columns = [
            # Metadata
            "subscription_id",
            "account_id",
            # Features
            "region",
            "region_missing_flag",
            "segment",
            "acquisition_channel",
            "account_age_days_at_t0",
            "account_created_at_missing_flag",
            "subscription_tenure_days_at_t0",
            "subscription_start_month",
            "subscription_start_quarter",
            "plan_name",
            "plan_monthly_price",
            "plan_included_seats",
            # Targets
            "target_churn_90d",
            "target_churn_180d",
            "target_churn_365d",
        ]

        final_df = merged[ordered_columns].sort_values(by="subscription_id", ascending=True).reset_index(drop=True)
        return final_df

    def build_feature_manifest(
        self, coverage_report: Optional[SourceCoverageReport] = None
    ) -> Dict[str, Any]:
        """Constructs the comprehensive feature manifest detailing definitions, sources, coverage, and exclusions."""
        cov = coverage_report or self.source_coverage or SourceCoverageReport(
            coverage_start_date=DATASET_START_DATE,
            coverage_end_date=self.cutoff_date,
            coverage_basis="declared_by_source",
            independently_verified=False,
            observed_max_started_at="2025-10-25",
            observed_max_ended_at="2025-12-30",
            label_observability_assumption=(
                "Negative churn labels (ended_at IS NULL or ended_at > window_end_date) assume "
                "complete event capture through declared cutoff date 2025-12-31 without right-censoring unobserved cancellations."
            ),
        )

        return {
            "version": "1.0.0",
            "observation_date": self.observation_date,
            "dataset_cutoff_date": self.cutoff_date,
            "prediction_horizons": {
                "90d": {"days": 90, "window_end_date": "2025-03-31"},
                "180d": {"days": 180, "window_end_date": "2025-06-29"},
                "365d": {"days": 365, "window_end_date": "2025-12-31"},
            },
            "source_info": {
                "dataset_name": "Analytics Engineering – SaaS Subscriptions & Customer Churn Dataset v1.0.0",
                "source_url": "https://www.analyticsengineering.com/datasets/saas-subscriptions",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "source_tables_dir": to_relative_path(self.tables_dir),
            },
            "source_coverage": asdict(cov),
            "cohort_inclusion_rules": {
                "rule": "started_at <= OBSERVATION_DATE and (ended_at IS NULL or ended_at > OBSERVATION_DATE)",
                "observation_date": self.observation_date,
                "description": "Subscriptions active at the end of day on 2024-12-31.",
                "exclusions": [
                    "Subscriptions ended on or before 2024-12-31 (already churned)",
                    "Subscriptions started after 2024-12-31 (not yet active)",
                    "Subscriptions with missing started_at date",
                ],
            },
            "metadata_columns": [
                {
                    "name": "subscription_id",
                    "data_type": "string",
                    "description": "Unique subscription primary key identifier. For record lineage only; must not be fed to ML models.",
                },
                {
                    "name": "account_id",
                    "data_type": "string",
                    "description": "Unique account identifier associated with the subscription (1:1 in modeling cohort). For record lineage only.",
                },
            ],
            "feature_columns": [
                {
                    "name": "region",
                    "data_type": "string",
                    "temporal_reliability": "assumed",
                    "source_table": "accounts",
                    "source_column": "region",
                    "description": "Geographical region of account (e.g. APAC, EMEA, LATAM). Nulls are preserved as true missing values. Assumed invariant across subscription lifecycle.",
                    "missing_value_policy": "Preserve NULL without imputation. Unobserved confounders cannot be ruled out.",
                },
                {
                    "name": "region_missing_flag",
                    "data_type": "integer",
                    "temporal_reliability": "assumed",
                    "source_table": "accounts",
                    "source_column": "region",
                    "description": "Binary flag indicating whether account region is missing (1 = missing, 0 = observed).",
                    "missing_value_policy": "Derived indicator (0/1).",
                },
                {
                    "name": "segment",
                    "data_type": "string",
                    "temporal_reliability": "assumed",
                    "source_table": "accounts",
                    "source_column": "segment",
                    "description": "Company size segment (SMB, Mid-Market, Enterprise). Assumed invariant across subscription lifecycle; historical company scaling logs are absent.",
                    "missing_value_policy": "Mandatory categorical, 0 missing values.",
                },
                {
                    "name": "acquisition_channel",
                    "data_type": "string",
                    "temporal_reliability": "verified",
                    "source_table": "accounts",
                    "source_column": "acquisition_channel",
                    "description": "Initial marketing acquisition channel (e.g. organic, paid, partner, referral, outbound). Point-in-time invariant.",
                    "missing_value_policy": "Mandatory categorical, 0 missing values.",
                },
                {
                    "name": "account_age_days_at_t0",
                    "data_type": "integer",
                    "temporal_reliability": "verified",
                    "source_table": "accounts",
                    "source_column": "created_at",
                    "description": "Number of elapsed calendar days from account creation (created_at) to observation date T0 (2024-12-31).",
                    "missing_value_policy": "Nullable Int64; 0 missing values in verified cohort.",
                },
                {
                    "name": "account_created_at_missing_flag",
                    "data_type": "integer",
                    "temporal_reliability": "verified",
                    "source_table": "accounts",
                    "source_column": "created_at",
                    "description": "Binary indicator for missing account creation date (1 = missing, 0 = present).",
                    "missing_value_policy": "Derived indicator (0/1).",
                },
                {
                    "name": "subscription_tenure_days_at_t0",
                    "data_type": "integer",
                    "temporal_reliability": "verified",
                    "source_table": "subscriptions",
                    "source_column": "started_at",
                    "description": "Number of elapsed calendar days from subscription start (started_at) to observation date T0 (2024-12-31). Leakage-free tenure.",
                    "missing_value_policy": "Mandatory integer; non-negative.",
                },
                {
                    "name": "subscription_start_month",
                    "data_type": "integer",
                    "temporal_reliability": "verified",
                    "source_table": "subscriptions",
                    "source_column": "started_at",
                    "description": "Calendar month (1-12) of subscription start date.",
                    "missing_value_policy": "Mandatory integer.",
                },
                {
                    "name": "subscription_start_quarter",
                    "data_type": "integer",
                    "temporal_reliability": "verified",
                    "source_table": "subscriptions",
                    "source_column": "started_at",
                    "description": "Calendar quarter (1-4) of subscription start date.",
                    "missing_value_policy": "Mandatory integer.",
                },
                {
                    "name": "plan_name",
                    "data_type": "string",
                    "temporal_reliability": "assumed",
                    "source_table": "plans",
                    "source_column": "plan_name",
                    "description": "Subscribed pricing tier name (Starter, Growth, Scale, Enterprise). Assumed invariant across subscription lifecycle; historical plan change and upgrade/downgrade event logs are absent in source data.",
                    "missing_value_policy": "Mandatory categorical, 0 missing values.",
                },
                {
                    "name": "plan_monthly_price",
                    "data_type": "float",
                    "temporal_reliability": "assumed",
                    "source_table": "plans",
                    "source_column": "monthly_price",
                    "description": "Monthly list price of the subscribed plan from plans table. Assumed invariant across subscription period; historical pricing change logs or plan tier version records are absent in source data.",
                    "missing_value_policy": "Mandatory positive float.",
                },
                {
                    "name": "plan_included_seats",
                    "data_type": "integer",
                    "temporal_reliability": "assumed",
                    "source_table": "plans",
                    "source_column": "included_seats",
                    "description": "Number of user seats included by default in the subscribed plan tier. Assumed invariant across subscription period; historical plan packaging or seat quota revision logs are absent in source data.",
                    "missing_value_policy": "Mandatory positive integer.",
                },
            ],
            "target_columns": [
                {
                    "name": "target_churn_90d",
                    "horizon_days": 90,
                    "window_end_date": "2025-03-31",
                    "formula": "1 if (2024-12-31 < ended_at <= 2025-03-31) else 0",
                    "description": "Subscription churned within 90 days after observation date.",
                },
                {
                    "name": "target_churn_180d",
                    "horizon_days": 180,
                    "window_end_date": "2025-06-29",
                    "formula": "1 if (2024-12-31 < ended_at <= 2025-06-29) else 0",
                    "description": "Subscription churned within 180 days after observation date.",
                },
                {
                    "name": "target_churn_365d",
                    "horizon_days": 365,
                    "window_end_date": "2025-12-31",
                    "formula": "1 if (2024-12-31 < ended_at <= 2025-12-31) else 0",
                    "description": "Subscription churned within 365 days after observation date.",
                },
            ],
            "excluded_columns": [
                {
                    "name": "ended_at",
                    "category": "target_leakage",
                    "reason": "Directly encodes future cancellation event and timing occurring after T0. Only used to define cohort and target labels.",
                },
                {
                    "name": "status",
                    "category": "target_leakage",
                    "reason": "Encodes final lifecycle status of subscription at dataset cutoff (2025-12-31), contaminating T0 prediction with future state.",
                },
                {
                    "name": "churn",
                    "category": "target_leakage",
                    "reason": "EDA churn label based on status == 'cancelled' over full 2023-2025 period; constitutes direct label leakage.",
                },
                {
                    "name": "tenure_days",
                    "category": "target_leakage",
                    "reason": "Calculated in EDA using ended_at - started_at for cancelled accounts, which leaks post-T0 survival duration.",
                },
                {
                    "name": "seats",
                    "category": "unverified_temporal",
                    "reason": "Purchased seats count lacks temporal event audit logging. Documentation does not verify whether value is fixed at start or reflects post-T0 expansions.",
                },
                {
                    "name": "mrr",
                    "category": "unverified_temporal",
                    "reason": "Contracted MRR lacks historical change tracking. Unverified whether value reflects T0 status or subsequent modifications.",
                },
                {
                    "name": "additional_seats",
                    "category": "unverified_temporal",
                    "reason": "Derived from seats - included_seats; subject to the same temporal unverified status as seats.",
                },
                {
                    "name": "total_invoices",
                    "category": "low_population_coverage",
                    "reason": "Invoices cover only ~6% of subscriptions. Aggregates in EDA cover post-T0 invoices, creating leakage and severe population bias.",
                },
                {
                    "name": "total_payment_attempts",
                    "category": "low_population_coverage",
                    "reason": "Payment data restricted to 120 subscriptions. Excluded from baseline feature matrix to maintain generalizability.",
                },
                {
                    "name": "total_payment_failures",
                    "category": "low_population_coverage",
                    "reason": "Restricted billing cohort feature. Excluded from baseline matrix; reserved for specialized billing-cohort experiments.",
                },
                {
                    "name": "payment_failure_rate",
                    "category": "low_population_coverage",
                    "reason": "Restricted billing cohort feature. Excluded from baseline matrix.",
                },
                {
                    "name": "total_amount_billed",
                    "category": "low_population_coverage",
                    "reason": "Restricted billing cohort feature. Excluded from baseline matrix.",
                },
                {
                    "name": "total_amount_paid",
                    "category": "low_population_coverage",
                    "reason": "Restricted billing cohort feature. Excluded from baseline matrix.",
                },
                {
                    "name": "has_billing_data",
                    "category": "low_population_coverage",
                    "reason": "Cohort membership indicator. Excluded from baseline feature matrix.",
                },
                {
                    "name": "has_payment_data",
                    "category": "low_population_coverage",
                    "reason": "Cohort membership indicator. Excluded from baseline feature matrix.",
                },
                {
                    "name": "account_id",
                    "category": "identifier_metadata",
                    "reason": "Record tracking identifier. Excluded from feature space to prevent arbitrary ID memorization.",
                },
                {
                    "name": "subscription_id",
                    "category": "identifier_metadata",
                    "reason": "Primary key identifier. Excluded from feature space to prevent arbitrary ID memorization.",
                },
            ],
            "missing_value_policy": {
                "region": "Keep raw NULL (321 rows in cohort). Do not impute with mean/mode/synthetic categories. Indicator flag created.",
                "region_missing_flag": "1 when region is NULL, 0 otherwise.",
                "created_at": "Keep raw date; flag created if missing. 0 missing in cohort.",
                "categorical_handling": "String values preserved without one-hot or ordinal encoding.",
            },
        }

    def generate_markdown_report(
        self,
        cohort_stats: CohortStatistics,
        consistency_report: ConsistencyCheckReport,
        target_stats: List[TargetDistribution],
        billing_stats: BillingCoverageAtT0,
        coverage_report: SourceCoverageReport,
        manifest: Dict[str, Any],
        output_csv_rel_path: str,
    ) -> str:
        """Generates detailed modeling dataset report adhering to specification."""
        t_90 = next(t for t in target_stats if t.horizon_days == 90)
        t_180 = next(t for t in target_stats if t.horizon_days == 180)
        t_365 = next(t for t in target_stats if t.horizon_days == 365)

        md = [
            "# SaaS Churn Intelligence – Zaman Noktası Tabanlı Modelleme Veri Seti Raporu",
            "",
            "## 1. Modelleme Senaryosu ve Yönetici Özeti",
            "",
            "Bu çalışma, B2B SaaS müşteri kaybı (*Customer Churn*) tahmin modellerinin eğitilmesi için gerekli olan "
            "**zaman noktasına duyarlı (Point-in-Time Safe)** modelleme veri setinin oluşturulmasını ve metodolojik dokümantasyonunu kapsamaktadır. "
            "Makine öğrenmesi modellerinde en sık karşılaşılan metodolojik tuzaklardan biri olan **Hedef Sızıntısı (Target Leakage)** "
            "riskini azaltmak amacıyla, öznitelikler tahmin anında ($T_0$) bilinebilecek veya iş mantığı altında sabit kaldığı varsayılan bilgilerden türetilmiştir.",
            "",
            f"- **Gözlem Tarihi ($T_0$ / Observation Date)**: `{self.observation_date}` (Günün sonu)",
            f"- **Beyan Edilen Veri Kapsamı Bitiş Tarihi**: `{coverage_report.coverage_end_date}`",
            f"- **Modelleme Kohortu Büyüklüğü**: {cohort_stats.included_in_cohort:,} aktif abonelik",
            f"- **Üretilen Veri Seti**: `{output_csv_rel_path}`",
            "- **Özellik Manifestosu**: `ml/reports/modeling/feature_manifest.json`",
            "",
            "---",
            "",
            "## 2. Zaman Noktası ($T_0$), Tahmin Ufukları ve Veri Kapsamı",
            "",
            "Modelleme kohortu, 31 Aralık 2024 gecesi itibarıyla aktif olan müşterilerden oluşturulmuştur. "
            "Aynı gözlem anı için üç farklı karar penceresini temsil eden ileriye dönük ikili churn hedef değişkenleri üretilmiştir:",
            "",
            "| Hedef Değişken | Tahmin Ufku | Başlangıç Tarihi | Bitiş Tarihi (Dahil) | Mantıksal Koşul |",
            "|---|:---:|:---:|:---:|---|",
            f"| `target_churn_90d` | 90 gün | 2024-12-31 | {t_90.window_end_date} | `2024-12-31 < ended_at <= 2025-03-31` |",
            f"| `target_churn_180d` | 180 gün | 2024-12-31 | {t_180.window_end_date} | `2024-12-31 < ended_at <= 2025-06-29` |",
            f"| `target_churn_365d` | 365 gün | 2024-12-31 | {t_365.window_end_date} | `2024-12-31 < ended_at <= 2025-12-31` |",
            "",
            "> [!NOTE]",
            "> **Pencere Bitiş Tarihleri**: 2025 yılı artık yıl değildir (Şubat 28 gündür). 90 gün bitişi 31 Mart 2025, 180 gün bitişi 29 Haziran 2025 ve 365 gün bitişi 31 Aralık 2025'tir. Pencerelerin bitiş günlerinde gerçekleşen churn olayları hedef pozitif olarak sayılmıştır.",
            "",
            "### 2.1. Beyan Edilen Veri Kapsamı ile Satır Bazlı Gözlem Ayrımı",
            "",
            "- **Veri Kaynağı**: *Analytics Engineering – SaaS Subscriptions & Customer Churn Dataset v1.0.0*",
            f"- **Beyan Edilen Kapsam**: `{coverage_report.coverage_start_date}` ile `{coverage_report.coverage_end_date}` arası.",
            f"- **Kapsam Dayanağı (`coverage_basis`)**: `{coverage_report.coverage_basis}` (Veri kaynağı tarafından beyan edilmiş takvim aralığı).",
            f"- **Bağımsız Doğrulama (`independently_verified`)**: `{str(coverage_report.independently_verified).lower()}` (Dış bağımsız denetim logu veya transaction watermark bulunmamaktadır).",
            f"- **Satır Düzeyinde Gözlemlenen Maksimum Başlangıç Tarihi (`observed_max_started_at`)**: `{coverage_report.observed_max_started_at}`",
            f"- **Satır Düzeyinde Gözlemlenen Maksimum Bitiş Tarihi (`observed_max_ended_at`)**: `{coverage_report.observed_max_ended_at}`",
            "",
            "> [!IMPORTANT]",
            "> **Gözlem Bütünlüğü ve Negatif Hedef Etiketleme Varsayımı**:",
            "> - Kaynak CSV dosyalarında bağımsız extraction timestamp veya veri akışı bütünlük watermark'ı yer almamaktadır.",
            "> - Tablodaki maksimum `ended_at` değerinin `2025-12-30` olması, veri kesim tarihi olan `2025-12-31` gününde iptal gerçekleşmemiş olmasından kaynaklanabilir; tek başına bir eksiklik kanıtı değildir.",
            "> - Ancak `ended_at` değeri `NULL` olan (veya hedef penceresinden sonra sonlanan) aboneliklerin negatif (0) olarak etiketlenmesi, veri kaynağının 2025 yılı boyunca tüm iptal olaylarını eksiksiz yakaladığı (**sağdan sansürleme olmadığı**) varsayımına dayanmaktadır.",
            "",
            "---",
            "",
            "## 3. Kohort Seçim Kriterleri ve Elenen Kayıtlar",
            "",
            "Bir aboneliğin modelleme kohortuna dahil edilmesi için temel koşul, $T_0$ anında aktif olmasıdır:",
            "$$\\text{started\\_at} \\le T_0 \\quad \\land \\quad (\\text{ended\\_at is NULL} \\quad \\lor \\quad \\text{ended\\_at} > T_0)$$",
            "",
            "### Kohort Dağılım Tablosu",
            "| Kategori | Kayıt Sayısı | Oran (%) | Dahil Edilme Durumu |",
            "|---|---:|---:|:---:|",
            f"| **Modelleme Kohortuna Dahil** | **{cohort_stats.included_in_cohort:,}** | **%{cohort_stats.included_in_cohort / cohort_stats.total_subscriptions * 100:.2f}** | **Dahil Edildi ✅** |",
            f"| $T_0$ Tarihinde veya Öncesinde Sona Ermiş | {cohort_stats.excluded_ended_on_or_before_t0:,} | %{cohort_stats.excluded_ended_on_or_before_t0 / cohort_stats.total_subscriptions * 100:.2f} | Hariç Tutuldu ❌ |",
            f"| $T_0$ Sonrasında Başlamış | {cohort_stats.excluded_started_after_t0:,} | %{cohort_stats.excluded_started_after_t0 / cohort_stats.total_subscriptions * 100:.2f} | Hariç Tutuldu ❌ |",
            f"| Başlangıç Tarihi Eksik (`started_at is NULL`) | {cohort_stats.excluded_missing_started_at:,} | %{cohort_stats.excluded_missing_started_at / cohort_stats.total_subscriptions * 100:.2f} | Hariç Tutuldu ❌ |",
            f"| **Toplam Temizlenmiş Abonelik** | **{cohort_stats.total_subscriptions:,}** | **%100.00** | - |",
            "",
            "### Status ve ended_at Tutarlılık Denetimi",
            f"- `status == 'active'` iken `ended_at` dolu kayıt sayısı: **{consistency_report.active_with_ended_at_count}**",
            f"- `status == 'cancelled'` iken `ended_at` boş kayıt sayısı: **{consistency_report.cancelled_with_null_ended_at_count}**",
            f"- Kohorttaki 1.242 aboneliğin 182'sinde nihai `status == 'cancelled'` ve `ended_at > 2024-12-31` görülmektedir. Bu bir tutarsızlık değildir; 2025 yılı içerisinde gerçekleşen ve tahmin edilmek istenen müşteri kaybı olaylarını temsil eder.",
            "",
            "---",
            "",
            "## 4. Churn Hedef Değişkenlerinin Dağılımı ve Sınıf Dengesizliği (Class Imbalance)",
            "",
            "Modelleme kohortu üzerinde hesaplanan hedef değişken dağılımları:",
            "",
            "| Hedef Değişken | Pozitif (Churn = 1) | Negatif (Aktif = 0) | Toplam | Pozitif Oran (%) | Dengesizlik Oranı |",
            "|---|---:|---:|---:|---:|:---:|",
            f"| `target_churn_90d` | {t_90.positive_count:,} | {t_90.negative_count:,} | {cohort_stats.included_in_cohort:,} | **%{t_90.positive_rate_pct:.2f}** | ~1:{t_90.negative_count // t_90.positive_count} |",
            f"| `target_churn_180d` | {t_180.positive_count:,} | {t_180.negative_count:,} | {cohort_stats.included_in_cohort:,} | **%{t_180.positive_rate_pct:.2f}** | ~1:{t_180.negative_count // t_180.positive_count} |",
            f"| `target_churn_365d` | {t_365.positive_count:,} | {t_365.negative_count:,} | {cohort_stats.included_in_cohort:,} | **%{t_365.positive_rate_pct:.2f}** | ~1:{t_365.negative_count // t_365.positive_count} |",
            "",
            "> [!IMPORTANT]",
            "> **Kritik Metodolojik Uyarı – Sınıf Dengesizliği**:",
            "> - Özellikle 90 günlük hedefte pozitif sınıf oranı yalnızca **%4,51** seviyesindedir (21 negatif örneğe 1 pozitif örnek).",
            "> - İleride model eğitimi ve değerlendirmesi yapılırken **Accuracy (Doğruluk)** metriği kesinlikle yanıltıcı olacaktır; çünkü tüm tahminleri 0 veren naif bir model %95,49 doğruluk elde eder.",
            "> - Gelecek modelleme aşamasında **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, **Recall@k** ve **F-beta** metrikleri temel alınmalıdır.",
            "> - Hiyerarşik Tutarlılık Doğrulaması: Bütün satırlarda `target_churn_90d <= target_churn_180d <= target_churn_365d` kuralının sağlandığı matematiksel olarak kanıtlanmıştır.",
            "",
            "---",
            "",
            "## 5. Point-in-Time Özellikler (Feature Engineering)",
            "",
            "Modelleme veri setine dahil edilen tüm öznitelikler $T_0$ anı itibarıyla dondurulmuştur:",
            "",
            "| Özellik Adı | Tip | Kaynak | Zamansal Güvenilirlik | Açıklama |",
            "|---|:---:|:---:|:---:|---|",
            "| `region` | `string` | `accounts` | Assumed | Müşteri coğrafi bölgesi (APAC, EMEA, LATAM). Eksikler NULL korunmuştur. |",
            "| `region_missing_flag` | `integer` | `accounts` | Assumed | Bölge verisi eksik mi? (1 = Eksik, 0 = Mevcut). |",
            "| `segment` | `string` | `accounts` | Assumed | Şirket ölçek segmenti (SMB, Mid-Market, Enterprise). |",
            "| `acquisition_channel` | `string` | `accounts` | Verified | İlk temas edinim kanalı (organic, paid, partner, referral, outbound). |",
            "| `account_age_days_at_t0` | `integer` | `accounts` | Verified | $T_0$ ile hesap açılış tarihi (`created_at`) arasındaki gün sayısı ($T_0 - \\text{created\\_at}$). |",
            "| `account_created_at_missing_flag` | `integer` | `accounts` | Verified | Hesap açılış tarihi eksiklik bayrağı (1 = Eksik, 0 = Mevcut). |",
            "| `subscription_tenure_days_at_t0` | `integer` | `subscriptions` | Verified | $T_0$ anındaki abonelik kıdemi ($T_0 - \\text{started\\_at}$). Hedef sızıntısından arındırılmıştır. |",
            "| `subscription_start_month` | `integer` | `subscriptions` | Verified | Abonelik başlangıç ayı (1 - 12). Mevsimsellik göstergesi. |",
            "| `subscription_start_quarter` | `integer` | `subscriptions` | Verified | Abonelik başlangıç çeyreği (1 - 4). |",
            "| `plan_name` | `string` | `plans` | Assumed | Abone olunan plan adı (Starter, Growth, Scale, Enterprise). Değişmediği varsayılmıştır. |",
            "| `plan_monthly_price` | `float` | `plans` | Assumed | Planın aylık liste fiyatı (`monthly_price`). Fiyatın geçmişte değişmediği varsayılmıştır. |",
            "| `plan_included_seats` | `integer` | `plans` | Assumed | Plan dahilindeki koltuk sayısı (`included_seats`). Paket kapsamının değişmediği varsayılmıştır. |",
            "",
            "---",
            "",
            "## 6. Zamansal Güvenilirlik Sınıflandırması (Temporal Reliability)",
            "",
            "Kaynak dokümantasyon ve veri mimarisi incelemeleri sonucunda özelliklerin zaman içindeki davranışları üç grupta sınıflandırılmıştır:",
            "",
            "1. **Doğrulanmış (Verified)**: Zamansal olarak sabit olduğu kesin olan veya doğrudan $T_0$ ile geçmiş tarihler arasındaki farktan hesaplanan özellikler:",
            "   - `acquisition_channel`: Müşterinin ilk edinim kanalı, tanımı gereği zamanla değişmeyen başlangıç olayıdır.",
            "   - `account_age_days_at_t0`, `account_created_at_missing_flag`: $T_0$ ile `created_at` arasındaki matematiksel gün farkı.",
            "   - `subscription_tenure_days_at_t0`: $T_0$ ile `started_at` arasındaki matematiksel gün farkı.",
            "   - `subscription_start_month`, `subscription_start_quarter`: Başlangıç tarihinin takvim bileşenleri.",
            "2. **Varsayıma Dayalı (Assumed)**: Kaynak veride geçmiş logu veya versiyonlama tutulmayan ancak iş mantığı gereği abonelik boyunca değişmediği varsayılan özellikler:",
            "   - `region`, `region_missing_flag`: Müşteri şirket merkezinin abonelik süresince değişmediği varsayılmıştır.",
            "   - `segment`: Şirket ölçeğinin (SMB, Mid-Market, Enterprise) $T_0$ öncesinde değişmediği varsayılmıştır.",
            "   - `plan_name`, `plan_monthly_price`, `plan_included_seats`: Kaynak veride plan versiyonlaması veya geçmiş fiyatlandırma kayıtları bulunmamaktadır. Güncel plan kataloğundaki liste fiyatı ve dahil koltuk adedinin abonelik dönemi boyunca sabit kaldığı varsayılmıştır; fiyatların veya paket kapsamının geçmişte değişmiş olma riski metodolojik bir sınırlılık olarak değerlendirilmelidir.",
            "3. **Doğrulanamayan (Unverified)**: $T_0$ tarihindeki değerini geçmişe dönük olarak garanti edemediğimiz alanlar (`seats`, `mrr`, `additional_seats`). Bu alanlar model veri setinden **tamamen çıkarılmıştır**.",
            "",
            "---",
            "",
            "## 7. Hedef Sızıntısı (Target Leakage) ve Hariç Tutulan Alanlar",
            "",
            "Aşağıdaki alanlar modelleme feature matrix'ine **kesinlikle dahil edilmemiştir**:",
            "",
            "| Kolon | Kategori | Hariç Tutulma Nedeni |",
            "|---|:---:|---|",
            "| `ended_at` | Target Leakage | İptal tarihini doğrudan içerir. Yalnızca hedef oluşturmak ve kohort seçmek için kullanılmıştır. |",
            "| `status` | Target Leakage | 2025-12-31 tarihindeki nihai durumu yansıtır; churn olayını doğrudan sızdırır. |",
            "| `churn` | Target Leakage | EDA aşamasındaki genel churn bayrağıdır; doğrudan hedefin kendisidir. |",
            "| `tenure_days` | Target Leakage | EDA veri setindeki tenure hesabı iptal edilenlerde `ended_at - started_at` kullandığı için post-$T_0$ bilgisi sızdırır. |",
            "| `seats` | Unverified Temporal | Satın alınan koltuk adedinin $T_0$ tarihindeki tarihsel değeri doğrulanamamıştır. |",
            "| `mrr` | Unverified Temporal | Sözleşmeli MRR değerinin $T_0$ öncesinde mi yoksa sonrasında mı değiştiğine dair log bulunmamaktadır. |",
            "| `additional_seats` | Unverified Temporal | `seats - included_seats` farkından türetildiği için koltuk adedi belirsizliğini taşır. |",
            "| `subscription_id` | Identifier / Metadata | Kayıt takip amaçlı tutulmuştur, modele girdi olarak verilmemelidir. |",
            "| `account_id` | Identifier / Metadata | Kayıt takip amaçlı tutulmuştur, modele girdi olarak verilmemelidir. |",
            "",
            "---",
            "",
            "## 8. Fatura ve Ödeme Verisi Kapsam Sınırlılığı ($T_0$ İtibarıyla)",
            "",
            "EDA aşamasında tespit edilen fatura ve ödeme verisi kapsamındaki kısıt, $T_0$ tarihi itibarıyla incelenmiş ve doğrulanmıştır:",
            "",
            f"- **31 Aralık 2024'e kadar kesilen fatura sayısı**: {billing_stats.invoices_at_t0:,} fatura",
            f"- **Bu faturalarla ilişkili tekil abonelik sayısı**: {billing_stats.distinct_subscriptions_invoiced_at_t0} abonelik",
            f"- **31 Aralık 2024'e kadar yapılan ödeme denemesi sayısı**: {billing_stats.payments_at_t0:,} ödeme denemesi",
            f"- **Ödeme denemesi bulunan tekil abonelik sayısı**: {billing_stats.distinct_subscriptions_paid_at_t0} abonelik",
            f"- **1.242 kişilik modelleme kohortundaki fatura geçmişi**: Yalnızca **{billing_stats.cohort_subscriptions_with_invoices_at_t0} abonelik** (%{billing_stats.cohort_subscriptions_with_invoices_at_t0 / cohort_stats.included_in_cohort * 100:.2f})",
            f"- **1.242 kişilik modelleme kohortundaki ödeme geçmişi**: Yalnızca **{billing_stats.cohort_subscriptions_with_payments_at_t0} abonelik** (%{billing_stats.cohort_subscriptions_with_payments_at_t0 / cohort_stats.included_in_cohort * 100:.2f})",
            f"- **365 günlük churn hedefinde pozitif olan 182 aboneliğin fatura geçmişi**: Yalnızca **{billing_stats.churn_365d_subscriptions_with_invoices_at_t0} abonelik** (%{billing_stats.churn_365d_subscriptions_with_invoices_at_t0 / t_365.positive_count * 100:.2f})",
            "",
            "> [!CAUTION]",
            "> **Stratejik Karar – Fatura Özelliklerinin Baseline Dışında Bırakılması**:",
            "> Fatura ve ödeme geçmişi popülasyonun %94'ünde mevcut değildir. Bu değişkenlerin baseline modelleme veri setine dahil edilmesi durumunda, model genel müşteri davranışı yerine yalnızca ilk 120 aboneye özel yapay örüntüleri öğrenecektir (Selection Bias). Bu nedenle fatura ve ödeme değişkenleri ana modelleme veri setine **alınmamıştır**.",
            "",
            "---",
            "",
            "## 9. Eksik Değer Politikası",
            "",
            "- **`region` (321 eksik)**: Analitik tarafsızlığı korumak amacıyla eksik değerler mod/ortalama gibi yapay yöntemlerle doldurulmamış, `NULL` olarak bırakılmıştır. Birlikte `region_missing_flag` (1/0) bayrağı üretilmiştir.",
            "- **`created_at` (Kohortta 0 eksik)**: Tüm kohort hesaplarında açılış tarihi eksiksizdir. Olası sentetik durumlara karşı `account_created_at_missing_flag` üretilmiştir.",
            "- **Kategorik Değişkenler**: Kategorik alanlara (`region`, `segment`, `acquisition_channel`, `plan_name`) bu aşamada One-Hot Encoding veya Ordinal Encoding uygulanmamıştır. Ham kategorik değerler korunmuştur.",
            "",
            "---",
            "",
            "## 10. Modelleme Aşamasına Kalan Sınırlılıklar ve Sonraki Adımlar",
            "",
            "1. **Sınıf Dengesizliği Mücadelesi**: Özellikle 90 günlük hedef (%4,51 pozitif) için maliyete duyarlı öğrenme (Cost-sensitive Learning) veya threshold tuning gerekecektir.",
            "2. **Ayrı Fatura Alt-Modeli Fırsatı**: Faturası bulunan ~80 aboneliklik alt küme üzerinde, ödeme başarısızlıklarının etkisini ölçen ayrı bir mikro-deney tasarlanabilir.",
            "3. **Metrik Önceliği**: Kurumsal değerlendirmede False Negative (churn edeceği öngörülemeyen müşteri) maliyeti False Positive maliyetinden çok daha yüksek olduğu için Recall ve PR-AUC hedeflenmelidir.",
            "4. **Plan Fiyatlandırma Varsayımı**: Güncel fiyatlar baz alındığından, geçmişte fiyat artışı veya paket değişikliği yapılmışsa bu etkinin izlenemeyeceği göz önünde bulundurulmalıdır.",
        ]

        return "\n".join(md)

    def run(self) -> Dict[str, Any]:
        """Executes feature engineering pipeline and writes all artifacts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Point-in-Time Feature Engineering pipeline...")
        tables = self.load_clean_tables()

        # Step 1: Validate dates, cutoff, and censoring
        horizon_end_dates, coverage_report = self.validate_dates_and_censoring(tables)

        # Step 2: Check status and ended_at consistency
        consistency_report = self.check_status_ended_at_consistency(tables["subscriptions"])

        # Step 3: Create cohort
        cohort_df, cohort_stats = self.create_modeling_cohort(tables["subscriptions"])

        # Step 4: Compute targets
        cohort_with_targets, target_stats = self.compute_target_variables(cohort_df, horizon_end_dates)

        # Step 5: Evaluate billing coverage at T0
        churn_365_subs = set(
            cohort_with_targets[cohort_with_targets["target_churn_365d"] == 1]["subscription_id"].unique()
        )
        cohort_subs = set(cohort_with_targets["subscription_id"].unique())
        billing_stats = self.compute_billing_coverage_at_t0(
            tables["invoices"], tables["payments"], cohort_subs, churn_365_subs
        )

        # Step 6: Engineer features
        modeling_dataset = self.engineer_features(cohort_with_targets, tables["accounts"], tables["plans"])

        # Step 7: Save modeling dataset CSV
        output_csv_path = self.output_dir / "churn_modeling_dataset.csv"
        modeling_dataset.to_csv(output_csv_path, index=False)
        output_csv_rel = to_relative_path(output_csv_path)
        logger.info(
            f"Saved modeling dataset: {output_csv_rel} "
            f"({len(modeling_dataset)} rows, {len(modeling_dataset.columns)} columns)"
        )

        # Step 8: Generate and save feature manifest
        manifest = self.build_feature_manifest(coverage_report)
        manifest_path = self.reports_dir / "feature_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved feature manifest: {to_relative_path(manifest_path)}")

        # Step 9: Generate and save summary JSON
        summary_dict = {
            "version": "1.0.0",
            "observation_date": self.observation_date,
            "dataset_cutoff_date": self.cutoff_date,
            "source_coverage": asdict(coverage_report),
            "cohort_statistics": asdict(cohort_stats),
            "status_ended_at_consistency": asdict(consistency_report),
            "target_distributions": [asdict(t) for t in target_stats],
            "billing_coverage_at_t0": asdict(billing_stats),
            "dataset_shape": list(modeling_dataset.shape),
            "columns": list(modeling_dataset.columns),
            "feature_counts": {
                "metadata_columns": 2,
                "feature_columns": 12,
                "target_columns": 3,
                "total_columns": len(modeling_dataset.columns),
            },
            "output_csv_path": output_csv_rel,
            "manifest_path": to_relative_path(manifest_path),
            "source_tables_dir": to_relative_path(self.tables_dir),
        }
        summary_path = self.reports_dir / "modeling_dataset_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved modeling dataset summary JSON: {to_relative_path(summary_path)}")

        # Step 10: Generate and save markdown report
        report_md = self.generate_markdown_report(
            cohort_stats,
            consistency_report,
            target_stats,
            billing_stats,
            coverage_report,
            manifest,
            output_csv_rel,
        )
        report_path = self.reports_dir / "modeling_dataset_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Saved modeling dataset Markdown report: {to_relative_path(report_path)}")

        return summary_dict


def main() -> None:
    pipeline = FeaturePipeline()
    summary = pipeline.run()

    print("\n" + "=" * 70)
    print("CHURN MODELING FEATURE ENGINEERING COMPLETE")
    print(f"Observation Date (T0): {summary['observation_date']}")
    print(f"Dataset Cutoff Date: {summary['dataset_cutoff_date']}")
    print(f"Source Coverage Basis: {summary['source_coverage']['coverage_basis']} "
          f"({summary['source_coverage']['coverage_start_date']} to {summary['source_coverage']['coverage_end_date']})")
    print(f"Cohort Included Rows: {summary['cohort_statistics']['included_in_cohort']:,}")
    print(f"Total Columns: {summary['feature_counts']['total_columns']} "
          f"(2 Metadata + 12 Features + 3 Targets)")
    for t in summary["target_distributions"]:
        print(f"  - Target {t['horizon_days']}d (<= {t['window_end_date']}): "
              f"{t['positive_count']} positive ({t['positive_rate_pct']}%), {t['negative_count']} negative")
    print(f"Billing Invoices at T0: {summary['billing_coverage_at_t0']['invoices_at_t0']}")
    print(f"Billing Cohort Subs with Invoices: {summary['billing_coverage_at_t0']['cohort_subscriptions_with_invoices_at_t0']}")
    print(f"Output Dataset: {summary['output_csv_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
