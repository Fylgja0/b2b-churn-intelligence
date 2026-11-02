"""
Unit and Negative Tests for data validation module.

Validates that DataValidator successfully passes on clean tables
AND actively catches violations across foreign keys, nulls, dates,
cohort coverage, and domain constraints.
"""

from pathlib import Path
import pandas as pd
import pytest

from ml.src.data.validation import DataValidator, ValidationSuiteResult


@pytest.fixture(scope="module")
def validator() -> DataValidator:
    return DataValidator()


@pytest.fixture(scope="module")
def validation_suite(validator: DataValidator) -> ValidationSuiteResult:
    return validator.validate_all()


# ============================================================================
# POSITIVE TESTS (Expected to pass on clean dataset)
# ============================================================================

def test_validation_suite_passes(validation_suite: ValidationSuiteResult) -> None:
    assert validation_suite.all_passed is True, f"Failed checks: {validation_suite.failed_checks}"
    assert validation_suite.failed_checks == 0
    assert validation_suite.total_checks >= 30


def test_primary_keys_uniqueness(validator: DataValidator) -> None:
    tables = validator.load_processed_tables()
    pk_checks = validator.validate_primary_keys(tables)
    for check in pk_checks:
        assert check.passed is True, f"PK uniqueness failed: {check.check_name}"


def test_foreign_keys_integrity(validator: DataValidator) -> None:
    tables = validator.load_processed_tables()
    fk_checks = validator.validate_foreign_keys(tables)
    for check in fk_checks:
        assert check.passed is True, f"FK integrity failed: {check.check_name}"
        assert check.details["null_count"] == 0
        assert check.details["unmatched_key_count"] == 0


def test_date_conversion_audit(validator: DataValidator) -> None:
    clean_tables = validator.load_processed_tables()
    raw_tables = validator.load_raw_tables()
    date_conv_checks = validator.validate_date_conversions(clean_tables, raw_tables)
    for check in date_conv_checks:
        assert check.passed is True, f"Date conversion audit failed: {check.check_name}"
        assert check.details["unparseable_failed_count"] == 0


def test_date_chronology(validator: DataValidator) -> None:
    tables = validator.load_processed_tables()
    date_checks = validator.validate_dates(tables)
    for check in date_checks:
        assert check.passed is True, f"Date consistency failed: {check.check_name}"


def test_numeric_ranges(validator: DataValidator) -> None:
    tables = validator.load_processed_tables()
    num_checks = validator.validate_numerics(tables)
    for check in num_checks:
        assert check.passed is True, f"Numeric range check failed: {check.check_name}"


def test_cross_table_cardinality(validator: DataValidator) -> None:
    tables = validator.load_processed_tables()
    coverage_checks = validator.validate_cross_table_coverage(tables)
    for check in coverage_checks:
        assert check.passed is True, f"Coverage check failed: {check.check_name}"

    # Verify specific details
    cardinality_check = next(c for c in coverage_checks if c.check_name == "coverage_account_subscription_1to1")
    assert cardinality_check.details["actual_unique_accounts"] == 2000
    assert cardinality_check.details["actual_unique_subscriptions"] == 2000

    cohort_check = next(c for c in coverage_checks if c.check_name == "coverage_invoice_cohort_match")
    assert cohort_check.details["actual_invoiced_subs_count"] == 120
    assert cohort_check.details["mismatch_count"] == 0


# ============================================================================
# NEGATIVE TESTS (Ensures validator catches corrupted data)
# ============================================================================

def test_negative_null_foreign_key(validator: DataValidator) -> None:
    """Negative Test 1: Catches NULL foreign key values."""
    tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    # Introduce NULL in subscriptions.account_id
    tables["subscriptions"].loc[0, "account_id"] = None

    fk_checks = validator.validate_foreign_keys(tables)
    sub_acc_check = next(c for c in fk_checks if c.check_name == "fk_subscriptions_account_id_to_accounts_account_id")

    assert sub_acc_check.passed is False, "Validator must fail when a foreign key is NULL"
    assert sub_acc_check.details["null_count"] == 1


def test_negative_unmatched_foreign_key(validator: DataValidator) -> None:
    """Negative Test 2: Catches unmatched / orphan foreign key values."""
    tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    # Introduce non-existent foreign key
    tables["subscriptions"].loc[0, "account_id"] = "A_NON_EXISTENT_999999"

    fk_checks = validator.validate_foreign_keys(tables)
    sub_acc_check = next(c for c in fk_checks if c.check_name == "fk_subscriptions_account_id_to_accounts_account_id")

    assert sub_acc_check.passed is False, "Validator must fail when a foreign key has no parent match"
    assert sub_acc_check.details["unmatched_key_count"] >= 1
    assert "A_NON_EXISTENT_999999" in sub_acc_check.details["unmatched_samples"]


def test_negative_account_subscription_cardinality_broken(validator: DataValidator) -> None:
    """Negative Test 3: Catches violation of 1:1 relationship between accounts and subscriptions."""
    tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    # Duplicate an account_id in subscriptions, breaking 1:1 mapping
    tables["subscriptions"].loc[1, "account_id"] = tables["subscriptions"].loc[0, "account_id"]

    coverage_checks = validator.validate_cross_table_coverage(tables)
    card_check = next(c for c in coverage_checks if c.check_name == "coverage_account_subscription_1to1")

    assert card_check.passed is False, "Validator must fail when account-subscription 1:1 mapping is broken"
    assert card_check.details["mismatch_count"] > 0


def test_negative_invoice_cohort_mismatch(validator: DataValidator) -> None:
    """Negative Test 4: Catches violation in expected invoice subscription cohort (S000001 - S000120)."""
    tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    # Introduce an invoice pointing to S000999 (outside the expected 120 cohort)
    tables["invoices"].loc[0, "subscription_id"] = "S000999"

    coverage_checks = validator.validate_cross_table_coverage(tables)
    cohort_check = next(c for c in coverage_checks if c.check_name == "coverage_invoice_cohort_match")

    assert cohort_check.passed is False, "Validator must fail when invoice cohort does not match expected subscriptions"
    assert cohort_check.details["mismatch_count"] > 0
    assert "S000999" in cohort_check.details["unexpected_samples"]


def test_negative_unparseable_date_conversion(validator: DataValidator) -> None:
    """Negative Test 5: Catches non-empty raw date that failed conversion / coerced to NaT."""
    clean_tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    raw_tables = {name: df.copy() for name, df in validator.load_raw_tables().items()}

    # Set raw date to a garbage string that was present in raw data
    target_pk = clean_tables["accounts"].loc[0, "account_id"]
    raw_tables["accounts"].loc[raw_tables["accounts"]["account_id"] == target_pk, "created_at"] = "NOT_A_VALID_DATE_STR"
    # Clean table has None / NaT because conversion failed
    clean_tables["accounts"].loc[clean_tables["accounts"]["account_id"] == target_pk, "created_at"] = None

    date_conv_checks = validator.validate_date_conversions(clean_tables, raw_tables)
    acc_date_check = next(c for c in date_conv_checks if c.check_name == "date_conversion_accounts_created_at")

    assert acc_date_check.passed is False, "Validator must fail when a non-empty raw date fails conversion"
    assert acc_date_check.details["unparseable_failed_count"] == 1


def test_negative_date_chronology_inconsistency(validator: DataValidator) -> None:
    """Negative Test 6: Catches chronological date inconsistency (ended_at < started_at)."""
    tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    # Set ended_at earlier than started_at on a cancelled subscription
    cancelled_mask = tables["subscriptions"]["status"] == "cancelled"
    idx = tables["subscriptions"][cancelled_mask].index[0]
    tables["subscriptions"].loc[idx, "started_at"] = "2025-06-01"
    tables["subscriptions"].loc[idx, "ended_at"] = "2024-01-01"

    date_checks = validator.validate_dates(tables)
    ended_started_check = next(c for c in date_checks if c.check_name == "date_order_subscriptions_started_vs_ended")

    assert ended_started_check.passed is False, "Validator must fail when ended_at is earlier than started_at"
    assert ended_started_check.details["violation_count"] >= 1


def test_negative_unexpected_categorical_value(validator: DataValidator) -> None:
    """Negative Test 7: Catches unexpected categorical domain value."""
    tables = {name: df.copy() for name, df in validator.load_processed_tables().items()}
    # Introduce invalid status in subscriptions
    tables["subscriptions"].loc[0, "status"] = "pending_review"

    cat_checks = validator.validate_categoricals(tables)
    status_check = next(c for c in cat_checks if c.check_name == "categorical_domain_subscriptions_status")

    assert status_check.passed is False, "Validator must fail on unexpected categorical value"
    assert "pending_review" in status_check.details["unexpected_values"]
