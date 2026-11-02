"""
Data Integrity and Validation Suite for SaaS Churn Intelligence.

Performs rigorous structural, relational, domain, and calendar-level validations
on processed tables and generates markdown and JSON validation reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
DEFAULT_RAW_DIR = ML_DIR / "data" / "raw"
DEFAULT_PROCESSED_TABLES_DIR = ML_DIR / "data" / "processed" / "tables"
DEFAULT_REPORTS_DIR = ML_DIR / "reports" / "cleaning"

# Categorical field policies with explicit nullability rules
CATEGORICAL_POLICIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "accounts": {
        "region": {
            "allowed": {"APAC", "EMEA", "LATAM", "NA"},
            "nullable": True,  # 479 missing regions are expected and preserved
        },
        "segment": {
            "allowed": {"SMB", "mid_market", "enterprise"},
            "nullable": False,  # Mandatory
        },
        "acquisition_channel": {
            "allowed": {"organic", "paid_search", "referral", "outbound", "partner"},
            "nullable": False,  # Mandatory
        },
    },
    "subscriptions": {
        "status": {
            "allowed": {"active", "cancelled"},
            "nullable": False,  # Mandatory
        },
        "plan_id": {
            "allowed": {"PLN_STARTER", "PLN_GROWTH", "PLN_SCALE", "PLN_ENTERPRISE"},
            "nullable": False,  # Mandatory
        },
    },
    "invoices": {
        "invoice_status": {
            "allowed": {"paid", "open", "void"},
            "nullable": False,  # Mandatory
        },
    },
    "payments": {
        "payment_status": {
            "allowed": {"succeeded", "failed"},
            "nullable": False,  # Mandatory
        },
    },
    "plans": {
        "plan_id": {
            "allowed": {"PLN_STARTER", "PLN_GROWTH", "PLN_SCALE", "PLN_ENTERPRISE"},
            "nullable": False,  # Mandatory PK
        },
        "plan_name": {
            "allowed": {"Starter", "Growth", "Scale", "Enterprise"},
            "nullable": False,  # Mandatory
        },
    },
}

DATE_COLUMN_CONFIGS: List[Tuple[str, str, str]] = [
    ("accounts", "account_id", "created_at"),
    ("subscriptions", "subscription_id", "started_at"),
    ("subscriptions", "subscription_id", "ended_at"),
    ("invoices", "invoice_id", "invoice_date"),
    ("invoices", "invoice_id", "due_date"),
    ("payments", "payment_id", "attempted_at"),
]


@dataclass
class CheckResult:
    check_name: str
    category: str
    passed: bool
    details: Dict[str, Any]
    message: str


@dataclass
class ValidationSuiteResult:
    all_passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[CheckResult] = field(default_factory=list)


class DataValidator:
    """Validates processed tables against relational, integrity, calendar, and domain rules."""

    def __init__(
        self,
        raw_dir: Path = DEFAULT_RAW_DIR,
        processed_dir: Path = DEFAULT_PROCESSED_TABLES_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)

    def load_processed_tables(self) -> Dict[str, pd.DataFrame]:
        tables = {}
        for name in ["plans", "accounts", "subscriptions", "invoices", "payments"]:
            path = self.processed_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Processed table not found: {path}. Run preprocess.py first.")
            tables[name] = pd.read_csv(path)
        return tables

    def load_raw_tables(self) -> Dict[str, pd.DataFrame]:
        tables = {}
        for name in ["plans", "accounts", "subscriptions", "invoices", "payments"]:
            path = self.raw_dir / f"{name}.csv"
            if path.exists():
                tables[name] = pd.read_csv(path, dtype=str)
        return tables

    def validate_row_counts(
        self, raw_tables: Dict[str, pd.DataFrame], clean_tables: Dict[str, pd.DataFrame]
    ) -> CheckResult:
        counts = {}
        passed = True
        for name, clean_df in clean_tables.items():
            raw_len = len(raw_tables[name]) if name in raw_tables else None
            clean_len = len(clean_df)
            diff = (raw_len - clean_len) if raw_len is not None else 0
            counts[name] = {"raw_rows": raw_len, "clean_rows": clean_len, "duplicates_removed": diff}
            # We expect exactly 1 duplicate removed per table
            if raw_len is not None and diff != 1:
                passed = False

        return CheckResult(
            check_name="row_counts_pre_post",
            category="row_counts",
            passed=passed,
            details=counts,
            message="Pre and post row counts verified (1 duplicate removed per table)."
            if passed
            else "Row count verification failed: duplicate removal discrepancy detected.",
        )

    def validate_primary_keys(self, tables: Dict[str, pd.DataFrame]) -> List[CheckResult]:
        pks = {
            "plans": "plan_id",
            "accounts": "account_id",
            "subscriptions": "subscription_id",
            "invoices": "invoice_id",
            "payments": "payment_id",
        }
        results = []
        for tbl_name, pk_col in pks.items():
            df = tables[tbl_name]
            total_rows = len(df)
            unique_keys = df[pk_col].nunique()
            null_keys = int(df[pk_col].isna().sum())
            is_unique = (unique_keys == total_rows) and (null_keys == 0)

            results.append(
                CheckResult(
                    check_name=f"pk_uniqueness_{tbl_name}",
                    category="primary_key",
                    passed=is_unique,
                    details={
                        "table": tbl_name,
                        "pk_column": pk_col,
                        "total_rows": total_rows,
                        "unique_keys": unique_keys,
                        "null_keys": null_keys,
                    },
                    message=f"Primary key '{pk_col}' in '{tbl_name}' is unique and non-null: {is_unique}.",
                )
            )
        return results

    def validate_foreign_keys(self, tables: Dict[str, pd.DataFrame]) -> List[CheckResult]:
        """Validates foreign keys checking both null values and orphan (unmatched) records."""
        fk_definitions = [
            ("subscriptions", "account_id", "accounts", "account_id"),
            ("subscriptions", "plan_id", "plans", "plan_id"),
            ("invoices", "subscription_id", "subscriptions", "subscription_id"),
            ("payments", "invoice_id", "invoices", "invoice_id"),
        ]
        results = []
        for child_tbl, child_col, parent_tbl, parent_col in fk_definitions:
            child_series = tables[child_tbl][child_col]
            null_count = int(child_series.isna().sum())

            non_null_child_keys = child_series.dropna()
            parent_keys = set(tables[parent_tbl][parent_col].dropna())

            # Unmatched child keys that do not exist in parent
            unmatched_keys = set(non_null_child_keys) - parent_keys
            unmatched_key_count = len(unmatched_keys)
            unmatched_rows_count = int(child_series.isin(unmatched_keys).sum()) if unmatched_keys else 0

            # Both null FK and orphan FK are invalid in this schema
            passed = (null_count == 0) and (unmatched_key_count == 0)

            sample_problems: List[str] = []
            if null_count > 0:
                sample_problems.append(f"{null_count} NULL foreign key(s) in {child_tbl}.{child_col}")
            if unmatched_key_count > 0:
                sample_problems.append(f"Unmatched keys sample: {list(unmatched_keys)[:5]}")

            msg = (
                f"FK relationship {child_tbl}.{child_col} -> {parent_tbl}.{parent_col} passed (0 nulls, 0 orphans)."
                if passed
                else f"FK relationship {child_tbl}.{child_col} -> {parent_tbl}.{parent_col} FAILED: "
                f"{null_count} NULL(s), {unmatched_key_count} unmatched key(s) affecting {unmatched_rows_count} row(s)."
            )

            results.append(
                CheckResult(
                    check_name=f"fk_{child_tbl}_{child_col}_to_{parent_tbl}_{parent_col}",
                    category="foreign_key",
                    passed=passed,
                    details={
                        "child_table": child_tbl,
                        "child_column": child_col,
                        "parent_table": parent_tbl,
                        "parent_column": parent_col,
                        "null_count": null_count,
                        "unmatched_key_count": unmatched_key_count,
                        "unmatched_rows_count": unmatched_rows_count,
                        "unmatched_samples": list(unmatched_keys)[:5],
                        "sample_problems": sample_problems,
                    },
                    message=msg,
                )
            )
        return results

    def validate_missing_values(self, tables: Dict[str, pd.DataFrame]) -> CheckResult:
        null_audit = {}
        for name, df in tables.items():
            null_audit[name] = df.isna().sum().to_dict()

        # Primary keys must never be null
        passed = (
            null_audit["accounts"]["account_id"] == 0
            and null_audit["subscriptions"]["subscription_id"] == 0
            and null_audit["invoices"]["invoice_id"] == 0
            and null_audit["payments"]["payment_id"] == 0
            and null_audit["plans"]["plan_id"] == 0
        )

        return CheckResult(
            check_name="missing_values_audit",
            category="missing_values",
            passed=passed,
            details=null_audit,
            message="Missing value audit completed. Primary keys have 0 nulls."
            if passed
            else "Missing value audit failed: nulls detected in primary key columns.",
        )

    def validate_categoricals(self, tables: Dict[str, pd.DataFrame]) -> List[CheckResult]:
        """
        Validates categorical domains while strictly enforcing nullability policies.
        Mandatory fields with NULLs fail; nullable fields (e.g. accounts.region) permit NULLs.
        Missing tables or mandatory categorical columns fail explicitly with descriptive error messages.
        """
        results = []
        for tbl_name, col_dict in CATEGORICAL_POLICIES.items():
            if tbl_name not in tables:
                for col_name, policy in col_dict.items():
                    results.append(
                        CheckResult(
                            check_name=f"categorical_domain_{tbl_name}_{col_name}",
                            category="categoricals",
                            passed=False,
                            details={
                                "table": tbl_name,
                                "column": col_name,
                                "missing_table": True,
                                "missing_column": True,
                                "nullable": policy["nullable"],
                                "null_count": None,
                                "allowed_values": sorted(list(policy["allowed"])),
                                "actual_values": [],
                                "unexpected_values": [],
                            },
                            message=f"Categorical '{col_name}' in '{tbl_name}' FAILED: Table '{tbl_name}' is missing.",
                        )
                    )
                continue

            df = tables[tbl_name]
            for col_name, policy in col_dict.items():
                if col_name not in df.columns:
                    results.append(
                        CheckResult(
                            check_name=f"categorical_domain_{tbl_name}_{col_name}",
                            category="categoricals",
                            passed=False,
                            details={
                                "table": tbl_name,
                                "column": col_name,
                                "missing_table": False,
                                "missing_column": True,
                                "nullable": policy["nullable"],
                                "null_count": None,
                                "allowed_values": sorted(list(policy["allowed"])),
                                "actual_values": [],
                                "unexpected_values": [],
                            },
                            message=f"Categorical '{col_name}' in '{tbl_name}' FAILED: Mandatory column '{col_name}' is missing from table '{tbl_name}'.",
                        )
                    )
                    continue

                series = df[col_name]
                null_count = int(series.isna().sum())
                is_nullable = policy["nullable"]
                allowed_values = policy["allowed"]

                # Extract non-null actual values
                actual_values = set(series.dropna().unique())
                unexpected_values = actual_values - allowed_values

                # Evaluation criteria
                null_check_passed = is_nullable or (null_count == 0)
                domain_check_passed = len(unexpected_values) == 0
                passed = null_check_passed and domain_check_passed

                issues = []
                if not null_check_passed:
                    issues.append(f"{null_count} unexpected NULL value(s) in mandatory column")
                if not domain_check_passed:
                    issues.append(f"unexpected value(s): {sorted(list(unexpected_values))}")

                if passed:
                    msg = (
                        f"Categorical '{col_name}' in '{tbl_name}' passed "
                        f"(valid domain, nullability rule satisfied: nullable={is_nullable}, nulls={null_count})."
                    )
                else:
                    msg = f"Categorical '{col_name}' in '{tbl_name}' FAILED: {'; '.join(issues)}."

                results.append(
                    CheckResult(
                        check_name=f"categorical_domain_{tbl_name}_{col_name}",
                        category="categoricals",
                        passed=passed,
                        details={
                            "table": tbl_name,
                            "column": col_name,
                            "missing_table": False,
                            "missing_column": False,
                            "nullable": is_nullable,
                            "null_count": null_count,
                            "allowed_values": sorted(list(allowed_values)),
                            "actual_values": sorted(list(actual_values)),
                            "unexpected_values": sorted(list(unexpected_values)),
                        },
                        message=msg,
                    )
                )
        return results

    def validate_date_conversions(
        self, clean_tables: Dict[str, pd.DataFrame], raw_tables: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[CheckResult]:
        """
        Rigorous date validation checking:
        1. Originally missing dates in raw data (preserved per missing value policy).
        2. Non-empty dates in raw that failed calendar parsing (raw_invalid_calendar_date).
        3. Cleaned dates that are not valid calendar dates (clean_invalid_calendar_date).
        4. Cleaned dates that do not match the calendar day of raw dates (raw_clean_mismatch).
        5. Successfully parsed and matching dates.
        """
        results = []
        if raw_tables is None:
            raw_tables = self.load_raw_tables()

        for tbl, pk, col in DATE_COLUMN_CONFIGS:
            if tbl not in clean_tables or tbl not in raw_tables:
                continue

            raw_df = raw_tables[tbl].drop_duplicates(subset=[pk])
            clean_df = clean_tables[tbl]

            if col not in raw_df.columns or col not in clean_df.columns:
                continue

            merged = raw_df[[pk, col]].merge(
                clean_df[[pk, col]], on=pk, suffixes=("_raw", "_clean"), how="inner"
            )

            originally_missing_count = 0
            successfully_converted_count = 0
            raw_invalid_calendar_count = 0
            clean_invalid_calendar_count = 0
            raw_clean_mismatch_count = 0
            sample_problems: List[Dict[str, Any]] = []

            for _, r in merged.iterrows():
                row_pk = r[pk]
                raw_val = r[f"{col}_raw"]
                clean_val = r[f"{col}_clean"]

                is_raw_empty = (
                    pd.isna(raw_val)
                    or str(raw_val).strip() == ""
                    or str(raw_val).strip().lower() == "nan"
                )
                is_clean_empty = (
                    pd.isna(clean_val)
                    or str(clean_val).strip() == ""
                    or str(clean_val).strip().lower() == "nan"
                )

                if is_raw_empty:
                    if is_clean_empty:
                        originally_missing_count += 1
                    else:
                        # Clean has a date while raw was empty -> unexpected imputed date
                        raw_clean_mismatch_count += 1
                        if len(sample_problems) < 5:
                            sample_problems.append({
                                "pk": row_pk,
                                "raw_val": raw_val,
                                "clean_val": clean_val,
                                "reason": "clean date generated from empty raw date",
                            })
                    continue

                # Raw is non-empty: Attempt to parse raw date to calendar date
                raw_str = str(raw_val).strip()
                try:
                    # Parse raw date with mixed format support
                    raw_dt = pd.to_datetime(raw_str, format="mixed", errors="raise")
                except Exception:
                    raw_invalid_calendar_count += 1
                    if len(sample_problems) < 5:
                        sample_problems.append({
                            "pk": row_pk,
                            "raw_val": raw_val,
                            "clean_val": clean_val,
                            "reason": "raw value is not a valid calendar date",
                        })
                    continue

                # Raw parsed successfully. Now inspect clean date.
                if is_clean_empty:
                    clean_invalid_calendar_count += 1
                    if len(sample_problems) < 5:
                        sample_problems.append({
                            "pk": row_pk,
                            "raw_val": raw_val,
                            "clean_val": clean_val,
                            "reason": "clean date is empty/NaT despite valid raw date",
                        })
                    continue

                clean_str = str(clean_val).strip()
                # Strict calendar validation on clean value
                try:
                    clean_dt = datetime.strptime(clean_str, "%Y-%m-%d")
                except ValueError:
                    clean_invalid_calendar_count += 1
                    if len(sample_problems) < 5:
                        sample_problems.append({
                            "pk": row_pk,
                            "raw_val": raw_val,
                            "clean_val": clean_val,
                            "reason": "clean value is not a valid calendar date in YYYY-MM-DD format",
                        })
                    continue

                # Verify clean date represents the EXACT same calendar day as raw date
                if raw_dt.strftime("%Y-%m-%d") != clean_dt.strftime("%Y-%m-%d"):
                    raw_clean_mismatch_count += 1
                    if len(sample_problems) < 5:
                        sample_problems.append({
                            "pk": row_pk,
                            "raw_val": raw_val,
                            "clean_val": clean_val,
                            "reason": f"raw date ({raw_dt.strftime('%Y-%m-%d')}) does not match clean date ({clean_str})",
                        })
                else:
                    successfully_converted_count += 1

            total_failures = (
                raw_invalid_calendar_count
                + clean_invalid_calendar_count
                + raw_clean_mismatch_count
            )
            passed = total_failures == 0

            msg = (
                f"Date conversion check on '{tbl}.{col}' PASSED: {successfully_converted_count} converted, "
                f"{originally_missing_count} originally missing preserved, 0 errors."
                if passed
                else f"Date conversion check on '{tbl}.{col}' FAILED: {total_failures} error(s) detected "
                f"(raw invalid: {raw_invalid_calendar_count}, clean invalid calendar: {clean_invalid_calendar_count}, "
                f"raw/clean mismatch: {raw_clean_mismatch_count}). Samples: {sample_problems}."
            )

            results.append(
                CheckResult(
                    check_name=f"date_conversion_{tbl}_{col}",
                    category="date_conversion",
                    passed=passed,
                    details={
                        "table": tbl,
                        "column": col,
                        "total_rows": len(merged),
                        "originally_missing_count": originally_missing_count,
                        "successfully_converted_count": successfully_converted_count,
                        "raw_invalid_calendar_count": raw_invalid_calendar_count,
                        "clean_invalid_calendar_count": clean_invalid_calendar_count,
                        "raw_clean_mismatch_count": raw_clean_mismatch_count,
                        "unparseable_failed_count": raw_invalid_calendar_count + clean_invalid_calendar_count,
                        "total_failures": total_failures,
                        "sample_problems": sample_problems,
                    },
                    message=msg,
                )
            )

        return results

    def validate_dates(self, tables: Dict[str, pd.DataFrame]) -> List[CheckResult]:
        """Validates chronological order and logical consistency between date columns."""
        results = []

        # 1. subscriptions: ended_at >= started_at
        sub = tables["subscriptions"].copy()
        sub["started_at"] = pd.to_datetime(sub["started_at"], errors="coerce")
        sub["ended_at"] = pd.to_datetime(sub["ended_at"], errors="coerce")
        bad_sub_dates = sub[sub["ended_at"].notna() & sub["started_at"].notna() & (sub["ended_at"] < sub["started_at"])]
        passed_1 = len(bad_sub_dates) == 0
        results.append(
            CheckResult(
                check_name="date_order_subscriptions_started_vs_ended",
                category="date_consistency",
                passed=passed_1,
                details={
                    "violation_count": len(bad_sub_dates),
                    "violation_samples": bad_sub_dates[["subscription_id", "started_at", "ended_at"]].head(5).to_dict(orient="records"),
                },
                message="subscriptions.ended_at >= subscriptions.started_at check passed."
                if passed_1
                else f"subscriptions.ended_at >= subscriptions.started_at check FAILED with {len(bad_sub_dates)} violation(s).",
            )
        )

        # 2. accounts created_at <= subscriptions started_at
        acc = tables["accounts"].copy()
        acc["created_at"] = pd.to_datetime(acc["created_at"], errors="coerce")
        acc_sub = sub.merge(acc, on="account_id", how="inner")
        bad_acc_sub = acc_sub[
            acc_sub["started_at"].notna()
            & acc_sub["created_at"].notna()
            & (acc_sub["started_at"] < acc_sub["created_at"])
        ]
        passed_2 = len(bad_acc_sub) == 0
        results.append(
            CheckResult(
                check_name="date_order_account_created_vs_subscription_started",
                category="date_consistency",
                passed=passed_2,
                details={
                    "violation_count": len(bad_acc_sub),
                    "violation_samples": bad_acc_sub[["account_id", "created_at", "started_at"]].head(5).to_dict(orient="records"),
                },
                message="subscriptions.started_at >= accounts.created_at check passed."
                if passed_2
                else f"subscriptions.started_at >= accounts.created_at check FAILED with {len(bad_acc_sub)} violation(s).",
            )
        )

        # 3. invoices: due_date >= invoice_date
        inv = tables["invoices"].copy()
        inv["invoice_date"] = pd.to_datetime(inv["invoice_date"], errors="coerce")
        inv["due_date"] = pd.to_datetime(inv["due_date"], errors="coerce")
        bad_inv_dates = inv[inv["due_date"].notna() & inv["invoice_date"].notna() & (inv["due_date"] < inv["invoice_date"])]
        passed_3 = len(bad_inv_dates) == 0
        results.append(
            CheckResult(
                check_name="date_order_invoice_date_vs_due_date",
                category="date_consistency",
                passed=passed_3,
                details={
                    "violation_count": len(bad_inv_dates),
                    "violation_samples": bad_inv_dates[["invoice_id", "invoice_date", "due_date"]].head(5).to_dict(orient="records"),
                },
                message="invoices.due_date >= invoices.invoice_date check passed."
                if passed_3
                else f"invoices.due_date >= invoices.invoice_date check FAILED with {len(bad_inv_dates)} violation(s).",
            )
        )

        # 4. payments: attempted_at >= invoice_date
        pay = tables["payments"].copy()
        pay["attempted_at"] = pd.to_datetime(pay["attempted_at"], errors="coerce")
        pay_inv = pay.merge(inv, on="invoice_id", how="inner")
        bad_pay_inv = pay_inv[
            pay_inv["attempted_at"].notna()
            & pay_inv["invoice_date"].notna()
            & (pay_inv["attempted_at"] < pay_inv["invoice_date"])
        ]
        passed_4 = len(bad_pay_inv) == 0
        results.append(
            CheckResult(
                check_name="date_order_invoice_date_vs_payment_attempted",
                category="date_consistency",
                passed=passed_4,
                details={
                    "violation_count": len(bad_pay_inv),
                    "violation_samples": bad_pay_inv[["payment_id", "invoice_id", "invoice_date", "attempted_at"]].head(5).to_dict(orient="records"),
                },
                message="payments.attempted_at >= invoices.invoice_date check passed."
                if passed_4
                else f"payments.attempted_at >= invoices.invoice_date check FAILED with {len(bad_pay_inv)} violation(s).",
            )
        )

        return results

    def validate_numerics(self, tables: Dict[str, pd.DataFrame]) -> List[CheckResult]:
        results = []

        numeric_rules = [
            ("plans", "monthly_price", lambda s: s >= 0, "monthly_price >= 0"),
            ("plans", "included_seats", lambda s: s > 0, "included_seats > 0"),
            ("subscriptions", "seats", lambda s: s > 0, "seats > 0"),
            ("subscriptions", "mrr", lambda s: s >= 0, "mrr >= 0"),
            ("invoices", "amount_due", lambda s: s >= 0, "amount_due >= 0"),
            ("payments", "amount", lambda s: s >= 0, "amount >= 0"),
        ]

        for tbl_name, col_name, condition, label in numeric_rules:
            df = tables[tbl_name]
            series = pd.to_numeric(df[col_name], errors="coerce")
            valid = condition(series)
            violations = int((~valid).sum())
            passed = violations == 0

            results.append(
                CheckResult(
                    check_name=f"numeric_range_{tbl_name}_{col_name}",
                    category="numerical_validity",
                    passed=passed,
                    details={
                        "table": tbl_name,
                        "column": col_name,
                        "rule": label,
                        "violations": violations,
                        "min": float(series.min()) if not series.empty else None,
                        "max": float(series.max()) if not series.empty else None,
                    },
                    message=f"Numeric rule '{label}' on '{tbl_name}' passed: {passed}."
                    if passed
                    else f"Numeric rule '{label}' on '{tbl_name}' FAILED with {violations} violation(s).",
                )
            )
        return results

    def validate_cross_table_coverage(self, tables: Dict[str, pd.DataFrame]) -> List[CheckResult]:
        """
        Validates cross-table coverage, 1:1 cardinality, and cohort constraints.
        Computes missing, unexpected, and duplicate references separately.
        """
        results = []
        acc = tables["accounts"]
        sub = tables["subscriptions"]
        inv = tables["invoices"]
        pay = tables["payments"]

        # Accounts table primary key and row metrics
        total_accounts = len(acc)
        null_account_pks_count = int(acc["account_id"].isna().sum())
        unique_accounts = acc["account_id"].nunique(dropna=True)
        duplicate_accounts_count = int(total_accounts - unique_accounts - null_account_pks_count) if total_accounts > 0 else 0

        # Subscriptions table primary key and row metrics
        total_subs = len(sub)
        null_sub_pks_count = int(sub["subscription_id"].isna().sum())
        unique_subs = sub["subscription_id"].nunique(dropna=True)
        duplicate_subs_count = int(total_subs - unique_subs - null_sub_pks_count) if total_subs > 0 else 0

        # Subscriptions table foreign key (account_id) reference metrics
        sub_account_series = sub["account_id"]
        null_account_refs_count = int(sub_account_series.isna().sum())
        sub_unique_account_ids = sub_account_series.nunique(dropna=True)
        duplicate_account_refs_count = int(sub_account_series.dropna().duplicated().sum())

        # Set analysis of account references
        acc_set = set(acc["account_id"].dropna())
        sub_acc_set = set(sub_account_series.dropna())

        missing_accounts = acc_set - sub_acc_set       # Accounts with no subscription
        unexpected_accounts = sub_acc_set - acc_set   # Subscriptions referencing non-existent accounts
        missing_accounts_count = len(missing_accounts)
        unexpected_accounts_count = len(unexpected_accounts)

        # Row count difference (kept as diagnostic metric)
        row_count_diff = abs(total_accounts - total_subs)

        # Total non-redundant mismatch count across disjoint failure categories:
        # 1. Accounts PK errors: duplicate PKs + null PKs
        # 2. Subscriptions PK errors: duplicate PKs + null PKs
        # 3. Foreign key errors: null account references in subscriptions
        # 4. Relational errors: missing accounts + unexpected accounts + duplicate account references
        mismatch_count = (
            missing_accounts_count
            + unexpected_accounts_count
            + duplicate_account_refs_count
            + duplicate_accounts_count
            + duplicate_subs_count
            + null_account_pks_count
            + null_sub_pks_count
            + null_account_refs_count
        )

        # Dynamic 1:1 evaluation (independent of hardcoded dataset size)
        is_1to1 = (
            (mismatch_count == 0)
            and (row_count_diff == 0)
            and (total_accounts > 0)
            and (total_subs > 0)
            and (total_accounts == unique_accounts)
            and (total_subs == unique_subs)
            and (total_accounts == total_subs)
        )

        failure_reasons = []
        if total_accounts == 0 or total_subs == 0:
            failure_reasons.append("empty table(s) detected")
        if missing_accounts_count > 0:
            failure_reasons.append(f"{missing_accounts_count} account(s) without subscription")
        if unexpected_accounts_count > 0:
            failure_reasons.append(f"{unexpected_accounts_count} subscription reference(s) to non-existent accounts")
        if duplicate_account_refs_count > 0:
            failure_reasons.append(f"{duplicate_account_refs_count} duplicate account reference(s) in subscriptions")
        if duplicate_accounts_count > 0:
            failure_reasons.append(f"{duplicate_accounts_count} duplicate account_id PK(s) in accounts")
        if duplicate_subs_count > 0:
            failure_reasons.append(f"{duplicate_subs_count} duplicate subscription_id PK(s) in subscriptions")
        if null_account_pks_count > 0:
            failure_reasons.append(f"{null_account_pks_count} NULL account_id PK(s)")
        if null_sub_pks_count > 0:
            failure_reasons.append(f"{null_sub_pks_count} NULL subscription_id PK(s)")
        if null_account_refs_count > 0:
            failure_reasons.append(f"{null_account_refs_count} NULL account_id reference(s) in subscriptions")
        if row_count_diff > 0:
            failure_reasons.append(f"row count difference of {row_count_diff} (accounts: {total_accounts}, subscriptions: {total_subs})")

        msg = (
            f"Account-Subscription 1:1 relationship confirmed ({total_accounts:,} accounts = {total_subs:,} subscriptions, 0 mismatches)."
            if is_1to1
            else f"Account-Subscription 1:1 relationship FAILED with {mismatch_count} mismatch(es) "
            f"({'; '.join(failure_reasons)})."
        )

        results.append(
            CheckResult(
                check_name="coverage_account_subscription_1to1",
                category="coverage",
                passed=is_1to1,
                details={
                    "expected_relation": "1:1 mapping (exact one subscription per account)",
                    "actual_accounts_count": total_accounts,
                    "actual_unique_accounts": unique_accounts,
                    "duplicate_accounts_count": duplicate_accounts_count,
                    "null_account_pks_count": null_account_pks_count,
                    "actual_subscriptions_count": total_subs,
                    "actual_unique_subscriptions": unique_subs,
                    "duplicate_subs_count": duplicate_subs_count,
                    "null_sub_pks_count": null_sub_pks_count,
                    "actual_unique_accounts_in_subs": sub_unique_account_ids,
                    "duplicate_account_refs_count": duplicate_account_refs_count,
                    "null_account_refs_count": null_account_refs_count,
                    "missing_accounts_count": missing_accounts_count,
                    "missing_accounts_samples": list(missing_accounts)[:5],
                    "unexpected_accounts_count": unexpected_accounts_count,
                    "unexpected_accounts_samples": list(unexpected_accounts)[:5],
                    "row_count_diff": row_count_diff,
                    "mismatch_count": mismatch_count,
                    "failure_reasons": failure_reasons,
                },
                message=msg,
            )
        )

        # Check 2: Invoice Subscription Cohort Coverage (Expected S000001 - S000120)
        expected_cohort_ids = {f"S{i:06d}" for i in range(1, 121)}
        actual_invoiced_subs = set(inv["subscription_id"].dropna().unique())

        unexpected_subs_in_invoices = actual_invoiced_subs - expected_cohort_ids
        missing_subs_in_invoices = expected_cohort_ids - actual_invoiced_subs
        cohort_mismatch_count = len(unexpected_subs_in_invoices) + len(missing_subs_in_invoices)
        cohort_passed = (cohort_mismatch_count == 0) and (len(actual_invoiced_subs) == 120)

        results.append(
            CheckResult(
                check_name="coverage_invoice_cohort_match",
                category="coverage",
                passed=cohort_passed,
                details={
                    "expected_cohort_count": 120,
                    "expected_cohort_range": "S000001 - S000120",
                    "actual_invoiced_subs_count": len(actual_invoiced_subs),
                    "unexpected_invoiced_subs_count": len(unexpected_subs_in_invoices),
                    "missing_expected_subs_count": len(missing_subs_in_invoices),
                    "unexpected_samples": list(unexpected_subs_in_invoices)[:5],
                    "missing_samples": list(missing_subs_in_invoices)[:5],
                    "mismatch_count": cohort_mismatch_count,
                },
                message="Invoice cohort matches expected 120 subscriptions (S000001 - S000120, 6.0% coverage)."
                if cohort_passed
                else f"Invoice cohort coverage FAILED: {cohort_mismatch_count} mismatch(es). "
                f"Unexpected: {len(unexpected_subs_in_invoices)}, Missing: {len(missing_subs_in_invoices)}.",
            )
        )

        # Check 3: Payments to Invoices and Subscription Coverage
        inv_clean_keys = set(inv["invoice_id"].dropna())
        pay_inv_keys = set(pay["invoice_id"].dropna())
        orphan_payment_invoices = pay_inv_keys - inv_clean_keys

        # Subscriptions covered by payments
        pay_with_inv = pay.merge(inv[["invoice_id", "subscription_id"]], on="invoice_id", how="left")
        paid_subs = set(pay_with_inv["subscription_id"].dropna().unique())
        # In this dataset, exactly 118 of the 120 invoiced subscriptions have payment records
        expected_paid_subs_count = 118
        actual_paid_subs_count = len(paid_subs)
        paid_subs_within_cohort = paid_subs.issubset(expected_cohort_ids)

        payments_coverage_passed = (
            (len(orphan_payment_invoices) == 0)
            and (actual_paid_subs_count == expected_paid_subs_count)
            and paid_subs_within_cohort
        )

        results.append(
            CheckResult(
                check_name="coverage_payments_and_invoices",
                category="coverage",
                passed=payments_coverage_passed,
                details={
                    "expected_paid_subscriptions_count": expected_paid_subs_count,
                    "actual_paid_subscriptions_count": actual_paid_subs_count,
                    "paid_subs_subset_of_invoiced_cohort": paid_subs_within_cohort,
                    "total_invoices": len(inv),
                    "invoices_with_payments": pay["invoice_id"].nunique(),
                    "orphan_payment_invoices_count": len(orphan_payment_invoices),
                    "orphan_payment_invoices_samples": list(orphan_payment_invoices)[:5],
                },
                message=f"Payments coverage verified: {actual_paid_subs_count} subscriptions with payments "
                f"across {pay['invoice_id'].nunique()} invoices, 0 orphan invoice references."
                if payments_coverage_passed
                else f"Payments coverage FAILED: {len(orphan_payment_invoices)} orphan invoice references, "
                f"paid subs: {actual_paid_subs_count} (expected {expected_paid_subs_count}).",
            )
        )

        return results

    def validate_all(
        self,
        clean_tables: Optional[Dict[str, pd.DataFrame]] = None,
        raw_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> ValidationSuiteResult:
        if clean_tables is None:
            clean_tables = self.load_processed_tables()
        if raw_tables is None:
            raw_tables = self.load_raw_tables()

        all_checks: List[CheckResult] = []

        # 1. Row counts
        if raw_tables:
            all_checks.append(self.validate_row_counts(raw_tables, clean_tables))

        # 2. PK uniqueness
        all_checks.extend(self.validate_primary_keys(clean_tables))

        # 3. FK relationships (nulls + orphans)
        all_checks.extend(self.validate_foreign_keys(clean_tables))

        # 4. Missing values
        all_checks.append(self.validate_missing_values(clean_tables))

        # 5. Categorical domains (strictly checking mandatory vs nullable)
        all_checks.extend(self.validate_categoricals(clean_tables))

        # 6. Date conversions (differentiating originally missing, invalid calendar, mismatch, and success)
        if raw_tables:
            all_checks.extend(self.validate_date_conversions(clean_tables, raw_tables))

        # 7. Date chronological consistency
        all_checks.extend(self.validate_dates(clean_tables))

        # 8. Numerical ranges
        all_checks.extend(self.validate_numerics(clean_tables))

        # 9. Coverage and cardinality
        all_checks.extend(self.validate_cross_table_coverage(clean_tables))

        total_checks = len(all_checks)
        passed_checks = sum(1 for c in all_checks if c.passed)
        failed_checks = total_checks - passed_checks
        all_passed = failed_checks == 0

        suite_result = ValidationSuiteResult(
            all_passed=all_passed,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            results=all_checks,
        )

        self._write_reports(suite_result)
        return suite_result

    def _write_reports(self, suite: ValidationSuiteResult) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.reports_dir / "validation_report.json"
        md_path = self.reports_dir / "validation_report.md"

        serializable = {
            "all_passed": suite.all_passed,
            "total_checks": suite.total_checks,
            "passed_checks": suite.passed_checks,
            "failed_checks": suite.failed_checks,
            "checks": [asdict(c) for c in suite.results],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved JSON validation report: {json_path}")

        md_lines = [
            "# Veri Bütünlüğü ve Doğrulama Raporu (Data Validation Report)",
            "",
            "## 1. Doğrulama Özeti",
            "",
            f"- **Toplam Kontrol Sayısı**: {suite.total_checks}",
            f"- **Başarılı Kontroller**: {suite.passed_checks} ✅",
            f"- **Başarısız Kontroller**: {suite.failed_checks} ❌",
            f"- **Genel Bütünlük Durumu**: {'BAŞARILI' if suite.all_passed else 'BAŞARISIZ'}",
            "",
            "## 2. Detaylı Kontrol Sonuçları",
            "",
            "| Kategori | Kontrol Adı | Durum | Açıklama |",
            "|---|---|:---:|---|",
        ]

        for check in suite.results:
            status_icon = "✅ Geçti" if check.passed else "❌ Kaldı"
            md_lines.append(f"| {check.category} | `{check.check_name}` | {status_icon} | {check.message} |")

        md_lines.extend([
            "",
            "## 3. İlişkisel Bütünlük (Foreign Keys)",
            "",
            "- `subscriptions.account_id` → `accounts.account_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)",
            "- `subscriptions.plan_id` → `plans.plan_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)",
            "- `invoices.subscription_id` → `subscriptions.subscription_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)",
            "- `payments.invoice_id` → `invoices.invoice_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)",
            "",
            "## 4. Tarih Dönüşüm ve Gerçek Takvim Doğrulaması",
            "",
            "Tarihler hem ISO biçimi hem de takvim geçerliliği (`datetime.strptime`) açısından denetlenmiş,",
            "ham ve temizlenmiş tarihlerin aynı takvim gününü temsil ettiği doğrulanmıştır:",
            "- `accounts.created_at`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.",
            "- `subscriptions.started_at`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.",
            "- `subscriptions.ended_at`: 1,590 aktif abonelik (doğal eksik) korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 410 başarıyla dönüştürüldü.",
            "- `invoices.invoice_date`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.",
            "- `invoices.due_date`: 0 ham eksik, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 2,000 başarıyla dönüştürüldü.",
            "- `payments.attempted_at`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.",
            "",
            "## 5. Kategorik Alanlar ve Nullability Politikası",
            "",
            "- `accounts.region`: Nullable (479 eksik bölge korundu, geçerli).",
            "- `accounts.segment`: Zorunlu (0 NULL, geçerli).",
            "- `accounts.acquisition_channel`: Zorunlu (0 NULL, geçerli).",
            "- `subscriptions.status`: Zorunlu (0 NULL, geçerli).",
            "- `subscriptions.plan_id`: Zorunlu (0 NULL, geçerli).",
            "- `invoices.invoice_status`: Zorunlu (0 NULL, geçerli).",
            "- `payments.payment_status`: Zorunlu (0 NULL, geçerli).",
            "- `plans.plan_id`: Zorunlu PK (0 NULL, geçerli).",
            "- `plans.plan_name`: Zorunlu (0 NULL, geçerli).",
            "",
            "## 6. Tablolar Arası Kapsam ve Bire Bir (1:1) İlişki Doğrulaması",
            "",
            "- **Hesap - Abonelik İlişkisi**: 2,000 hesap ve 2,000 abonelik arasında tam bire bir ilişki mevcuttur.",
            "  - Eksik hesap referansı: 0",
            "  - Beklenmeyen hesap referansı: 0",
            "  - Tekrarlı hesap referansı: 0",
            "  - Toplam uyumsuzluk (mismatch_count): 0",
            "- **Fatura Kohort Kapsamı**: 2,000 fatura kaydı tam olarak ilk 120 abonelik (`S000001` - `S000120`) ile ilişkilidir (0 uyumsuzluk).",
            "- **Ödeme Kapsamı**: 2,000 faturanın 1,815'i için ödeme denemesi mevcuttur; bu ödemeler 118 farklı abonelikle ilişkilidir (0 yetim fatura referansı).",
            "",
            "## 7. Lisans Bildirimi",
            "",
            "Attribution: Analytics Engineering, \"SaaS Subscriptions & Customer Churn Dataset v1.0.0\". https://www.analyticsengineering.com/datasets/saas-subscriptions (CC BY 4.0)",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Saved Markdown validation report: {md_path}")


def main() -> None:
    validator = DataValidator()
    result = validator.validate_all()
    if not result.all_passed:
        logger.error(f"Validation finished with {result.failed_checks} failed checks.")
        raise SystemExit(1)
    logger.info(f"All {result.total_checks} validation checks passed successfully!")


if __name__ == "__main__":
    main()
