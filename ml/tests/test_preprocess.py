"""
Unit tests for data preprocessing module.
"""

from pathlib import Path
import pandas as pd
import pytest

from ml.src.data.preprocess import (
    DataPreprocessor,
    DEFAULT_RAW_DIR,
    DEFAULT_PROCESSED_TABLES_DIR,
    TABLE_METADATA,
)


@pytest.fixture(scope="module")
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor()


def test_raw_files_exist(preprocessor: DataPreprocessor) -> None:
    for table_name, meta in TABLE_METADATA.items():
        file_path = preprocessor.raw_dir / meta["file_name"]
        assert file_path.exists(), f"Raw file {file_path} is missing"


def test_duplicate_removal_and_row_counts(preprocessor: DataPreprocessor) -> None:
    results = preprocessor.run()
    expected_clean_rows = {
        "plans": 4,
        "accounts": 2000,
        "subscriptions": 2000,
        "invoices": 2000,
        "payments": 2000,
    }
    for table_name, expected_count in expected_clean_rows.items():
        metric = results[table_name]
        assert metric.duplicates_removed == 1, f"{table_name} should have exactly 1 duplicate removed"
        assert metric.clean_rows == expected_count, f"{table_name} should have {expected_count} clean rows"
        assert metric.pk_is_unique is True, f"PK in {table_name} must be unique"


def test_whitespace_and_casing_cleaned(preprocessor: DataPreprocessor) -> None:
    # Read processed accounts
    accounts_clean = pd.read_csv(preprocessor.processed_dir / "accounts.csv")
    assert not accounts_clean["region"].str.startswith(" ").any()
    assert not accounts_clean["region"].str.endswith(" ").any()

    # Read processed subscriptions
    subs_clean = pd.read_csv(preprocessor.processed_dir / "subscriptions.csv")
    assert set(subs_clean["status"].unique()) == {"active", "cancelled"}

    # Read processed invoices
    invoices_clean = pd.read_csv(preprocessor.processed_dir / "invoices.csv")
    assert set(invoices_clean["invoice_status"].unique()) == {"paid", "open", "void"}

    # Read processed payments
    payments_clean = pd.read_csv(preprocessor.processed_dir / "payments.csv")
    assert set(payments_clean["payment_status"].unique()) == {"succeeded", "failed"}


def test_missing_values_preserved_policy(preprocessor: DataPreprocessor) -> None:
    accounts_clean = pd.read_csv(preprocessor.processed_dir / "accounts.csv")
    subs_clean = pd.read_csv(preprocessor.processed_dir / "subscriptions.csv")

    # accounts.region missing values must be preserved (~479 rows)
    region_nulls = accounts_clean["region"].isna().sum()
    assert region_nulls == 479, f"Expected 479 nulls in accounts.region, got {region_nulls}"

    # subscriptions.ended_at missing values must be preserved for active subscriptions (1591 raw - 1 duplicate = 1590 clean)
    ended_at_nulls = subs_clean["ended_at"].isna().sum()
    assert ended_at_nulls == 1590, f"Expected 1590 nulls in subscriptions.ended_at, got {ended_at_nulls}"

    # All cancelled subscriptions must have non-null ended_at
    cancelled_subs = subs_clean[subs_clean["status"] == "cancelled"]
    assert cancelled_subs["ended_at"].notna().all(), "All cancelled subscriptions must have ended_at date"


def test_date_iso_format(preprocessor: DataPreprocessor) -> None:
    accounts_clean = pd.read_csv(preprocessor.processed_dir / "accounts.csv")
    # Date should match YYYY-MM-DD pattern (or be NaN)
    dates = accounts_clean["created_at"].dropna()
    assert not dates.str.contains("/").any(), "Slash dates should be converted to ISO format"
    assert dates.str.match(r"^\d{4}-\d{2}-\d{2}$").all(), "All non-null dates should match YYYY-MM-DD"
