"""
Unit tests for dataset building module.
"""

from pathlib import Path
import pandas as pd
import pytest

from ml.src.data.build_dataset import DatasetBuilder, DatasetBuildSummary


@pytest.fixture(scope="module")
def builder() -> DatasetBuilder:
    return DatasetBuilder()


@pytest.fixture(scope="module")
def built_dataset(builder: DatasetBuilder) -> tuple[pd.DataFrame, DatasetBuildSummary]:
    return builder.build_merged_dataset()


def test_merged_dataset_grain_and_row_count(built_dataset: tuple[pd.DataFrame, DatasetBuildSummary]) -> None:
    df, summary = built_dataset
    assert summary.total_merged_rows == 2000, "Dataset must retain exactly 2000 rows"
    assert len(df) == 2000
    assert df["subscription_id"].is_unique, "subscription_id must be strictly unique"
    assert df["account_id"].is_unique, "account_id must be strictly unique"
    assert summary.is_one_to_one_account_subscription is True


def test_no_row_fanout_from_invoices_and_payments(built_dataset: tuple[pd.DataFrame, DatasetBuildSummary]) -> None:
    df, summary = built_dataset
    # Despite 2000 invoices and 2000 payments, dataset row count must remain 2000
    assert len(df) == 2000
    assert summary.subscriptions_with_invoices == 120
    assert summary.subscription_invoice_coverage_pct == 6.0


def test_churn_target_distribution(built_dataset: tuple[pd.DataFrame, DatasetBuildSummary]) -> None:
    df, summary = built_dataset
    assert "churn" in df.columns
    assert set(df["churn"].unique()) == {0, 1}
    cancelled_count = (df["status"] == "cancelled").sum()
    assert (df["churn"] == 1).sum() == cancelled_count
    assert 20.0 <= summary.churn_rate_pct <= 21.0


def test_engineered_features(built_dataset: tuple[pd.DataFrame, DatasetBuildSummary]) -> None:
    df, _ = built_dataset
    assert "tenure_days" in df.columns
    assert "additional_seats" in df.columns
    assert "has_billing_data" in df.columns
    assert "has_payment_data" in df.columns

    # Non-null tenures must be positive
    valid_tenure = df["tenure_days"].dropna()
    assert (valid_tenure > 0).all()

    # Additional seats must be >= 0
    assert (df["additional_seats"] >= 0).all()
