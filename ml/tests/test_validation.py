"""
Unit tests for data validation module.
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


def test_validation_suite_passes(validation_suite: ValidationSuiteResult) -> None:
    assert validation_suite.all_passed is True, f"Failed checks: {validation_suite.failed_checks}"
    assert validation_suite.failed_checks == 0
    assert validation_suite.total_checks >= 25


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
    coverage_check = validator.validate_cross_table_coverage(tables)
    assert coverage_check.passed is True
    details = coverage_check.details
    assert details["account_subscription_is_1_to_1"] is True
    assert details["subscriptions_with_invoices"] == 120
    assert details["subscription_invoice_coverage_pct"] == 6.0
