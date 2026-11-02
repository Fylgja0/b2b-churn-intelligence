"""
Unit Tests for Exploratory Data Analysis (EDA) module and Chi-Square statistical validations.

Tests verify:
1. Chi-square independence tests return valid statistics, dof, and p-values in [0, 1].
2. Cochran's expected cell frequency assumptions (E_ij >= 5) are evaluated accurately.
3. Degenerate, zero-variance, or empty tables are handled gracefully without runtime exceptions.
4. Core EDA metrics and cohorts match verified dataset dimensions.
"""

from pathlib import Path
import pandas as pd
import pytest

from ml.src.eda.eda import SaasChurnEDA


@pytest.fixture(scope="module")
def eda_analyzer() -> SaasChurnEDA:
    return SaasChurnEDA()


@pytest.fixture(scope="module")
def eda_data(eda_analyzer: SaasChurnEDA) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return eda_analyzer.load_data()


@pytest.fixture(scope="module")
def eda_dataset(eda_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    return eda_data[0]


@pytest.fixture(scope="module")
def clean_tables(eda_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    return eda_data[1]


@pytest.fixture(scope="module")
def eda_metrics(
    eda_analyzer: SaasChurnEDA, eda_dataset: pd.DataFrame, clean_tables: dict[str, pd.DataFrame]
) -> dict:
    return eda_analyzer.compute_metrics(eda_dataset, clean_tables)


def test_chi2_statistics_and_pvalues_valid(eda_metrics: dict) -> None:
    """Verifies that chi2 statistics, degrees of freedom, and p-values are mathematically valid."""
    assert "region_independence_tests" in eda_metrics
    tests = eda_metrics["region_independence_tests"]

    tested_columns = ["churn", "segment", "acquisition_channel", "plan_name"]
    for col in tested_columns:
        assert col in tests, f"Missing chi-square test result for column: {col}"
        res = tests[col]
        assert res["status"] == "success", f"Chi-square test failed for {col}: {res.get('error')}"
        assert isinstance(res["chi2_statistic"], float)
        assert res["chi2_statistic"] >= 0.0, f"Chi2 statistic cannot be negative: {res['chi2_statistic']}"
        assert isinstance(res["degrees_of_freedom"], int)
        assert res["degrees_of_freedom"] >= 1
        assert isinstance(res["p_value"], float)
        assert 0.0 <= res["p_value"] <= 1.0, f"P-value must be bounded in [0, 1]: {res['p_value']}"


def test_cochran_expected_frequency_assumptions(eda_metrics: dict) -> None:
    """Verifies Cochran's rule that expected cell frequencies are >= 5."""
    tests = eda_metrics["region_independence_tests"]

    for col, res in tests.items():
        assert "min_expected_frequency" in res
        assert "cells_expected_lt_5" in res
        assert "assumptions_met" in res

        # On the 2,000-row B2B dataset, min expected cell frequency is well above 5 (min is ~24.9 for Enterprise plan)
        assert res["min_expected_frequency"] >= 5.0, f"Expected cell frequency < 5 for {col}: {res['min_expected_frequency']}"
        assert res["cells_expected_lt_5"] == 0, f"Found cells with expected count < 5 in {col}"
        assert res["assumptions_met"] is True, f"Assumptions should be satisfied for {col}"


def test_chi2_error_handling_degenerate_table(eda_analyzer: SaasChurnEDA) -> None:
    """Verifies that degenerate contingency tables (e.g. constant value, empty) handle gracefully without crash."""
    # Create degenerate dataset where a feature has only 1 unique constant value
    degenerate_df = pd.DataFrame({
        "account_id": ["A1", "A2", "A3"],
        "region": [None, None, None],
        "churn": [False, False, False],
        "segment": ["Mid-Market", "Mid-Market", "Mid-Market"],
        "acquisition_channel": ["Outbound", "Outbound", "Outbound"],
        "plan_name": ["Growth", "Growth", "Growth"],
        "status": ["active", "active", "active"],
        "mrr": [100.0, 100.0, 100.0],
        "tenure_days": [30, 30, 30],
        "has_billing_data": [False, False, False],
        "total_payment_failures": [0, 0, 0],
    })

    metrics = eda_analyzer.compute_metrics(degenerate_df, tables={})
    assert "region_independence_tests" in metrics
    tests = metrics["region_independence_tests"]

    for col in ["churn", "segment", "acquisition_channel", "plan_name"]:
        assert col in tests
        res = tests[col]
        # Should not raise exception, status must be cannot_compute or error
        assert res["status"] in ("cannot_compute", "error")
        assert res["assumptions_met"] is False
        assert res["p_value"] is None


def test_eda_metrics_consistency(eda_metrics: dict, eda_dataset: pd.DataFrame) -> None:
    """Verifies mathematical consistency of summary metrics."""
    assert eda_metrics["total_accounts"] == 2000
    assert eda_metrics["active_subscriptions"] == 1590
    assert eda_metrics["churned_subscriptions"] == 410
    assert eda_metrics["active_subscriptions"] + eda_metrics["churned_subscriptions"] == 2000
    assert eda_metrics["overall_churn_rate_pct"] == 20.5
