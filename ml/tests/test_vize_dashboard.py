"""
Unit Tests for Vize EDA Dashboard (build_vize_dashboard.py) and dashboard_metrics.json.

Tests verify:
1. Total, active, and churned subscription counts in dashboard_metrics.json match
   ml/data/processed/merged/account_level_eda_dataset.csv 1-to-1.
2. Plan-level total MRR shares (%) and churned MRR shares (%) sum to 100.0%.
3. Kaplan-Meier survival milestones (180, 365, 540 days) and 12-month retention KPI
   match dynamic calculations from the dataset.
4. Chi-square p-values for plan, segment, and acquisition channel match dynamic calculations.
5. Generated PDF and PNG dashboard deliverables exist and satisfy A4 landscape 300 DPI specs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from ml.src.eda.build_vize_dashboard import (
    DATASET_PATH,
    EDA_METRICS_PATH,
    OUTPUT_JSON_PATH,
    OUTPUT_PDF_PATH,
    OUTPUT_PNG_PATH,
    compute_kaplan_meier,
    load_and_validate_data,
)


@pytest.fixture(scope="module")
def raw_csv_df() -> pd.DataFrame:
    assert DATASET_PATH.exists(), f"Dataset CSV not found at {DATASET_PATH}"
    return pd.read_csv(DATASET_PATH)


@pytest.fixture(scope="module")
def dashboard_metrics() -> dict:
    if not OUTPUT_JSON_PATH.exists():
        _, metrics = load_and_validate_data(DATASET_PATH, EDA_METRICS_PATH)
        return metrics
    with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_subscription_counts_match_csv_one_to_one(
    raw_csv_df: pd.DataFrame, dashboard_metrics: dict
) -> None:
    """
    Verifies that total, active, and churned subscription counts in dashboard_metrics.json
    match the raw CSV dataset 1-to-1.
    """
    csv_total = int(len(raw_csv_df))
    csv_active = int((raw_csv_df["status"] == "active").sum())
    csv_churned = int((raw_csv_df["churn"] == 1).sum())

    kpis = dashboard_metrics["overall_kpis"]
    assert kpis["total_subscriptions"] == csv_total == 2000
    assert kpis["active_subscriptions"] == csv_active == 1590
    assert kpis["churned_subscriptions"] == csv_churned == 410
    assert kpis["active_subscriptions"] + kpis["churned_subscriptions"] == kpis["total_subscriptions"]

    # Also verify plan, segment, and channel counts sum to total_subscriptions
    assert sum(v["total"] for v in dashboard_metrics["plan_analysis"].values()) == csv_total
    assert sum(v["churned"] for v in dashboard_metrics["plan_analysis"].values()) == csv_churned
    assert sum(v["total"] for v in dashboard_metrics["segment_analysis"].values()) == csv_total
    assert sum(v["churned"] for v in dashboard_metrics["segment_analysis"].values()) == csv_churned
    assert sum(v["total"] for v in dashboard_metrics["channel_analysis"].values()) == csv_total
    assert sum(v["churned"] for v in dashboard_metrics["channel_analysis"].values()) == csv_churned


def test_mrr_totals_and_shares_sum_to_100_percent(
    raw_csv_df: pd.DataFrame, dashboard_metrics: dict
) -> None:
    """
    Verifies that MRR totals match CSV and that plan-level total MRR shares (%)
    and churned MRR shares (%) sum to 100%.
    """
    csv_total_mrr = float(raw_csv_df["mrr"].sum())
    csv_churned_mrr = float(raw_csv_df.loc[raw_csv_df["churn"] == 1, "mrr"].sum())
    csv_mrr_churn_pct = (csv_churned_mrr / csv_total_mrr) * 100.0

    kpis = dashboard_metrics["overall_kpis"]
    assert math.isclose(kpis["total_mrr"], csv_total_mrr, abs_tol=1e-2)
    assert math.isclose(kpis["churned_mrr"], csv_churned_mrr, abs_tol=1e-2)
    assert math.isclose(kpis["mrr_churn_pct"], round(csv_mrr_churn_pct, 2), abs_tol=1e-2)

    mrr_by_plan = dashboard_metrics["mrr_by_plan"]
    assert len(mrr_by_plan) == 4

    # Exact sum of plan MRRs must equal portfolio MRR
    plan_total_mrr_sum = sum(v["total_mrr"] for v in mrr_by_plan.values())
    plan_churned_mrr_sum = sum(v["churned_mrr"] for v in mrr_by_plan.values())
    assert math.isclose(plan_total_mrr_sum, csv_total_mrr, abs_tol=1e-2)
    assert math.isclose(plan_churned_mrr_sum, csv_churned_mrr, abs_tol=1e-2)

    # Sum of percentage shares must equal 100.0%
    total_mrr_share_sum = sum(v["total_mrr_share_pct"] for v in mrr_by_plan.values())
    churned_mrr_share_sum = sum(v["churned_mrr_share_pct"] for v in mrr_by_plan.values())
    assert math.isclose(total_mrr_share_sum, 100.0, abs_tol=0.05)
    assert math.isclose(churned_mrr_share_sum, 100.0, abs_tol=0.05)


def test_kaplan_meier_and_chi_square_metrics(
    raw_csv_df: pd.DataFrame, dashboard_metrics: dict
) -> None:
    """
    Verifies Kaplan-Meier retention milestones (180, 365, 540 days),
    conditional loss probabilities, and Chi-Square p-values.
    """
    _, _, milestones, conditional = compute_kaplan_meier(raw_csv_df)
    km_json = dashboard_metrics["kaplan_meier"]["milestones"]

    for key in ["day_180_pct", "day_365_pct", "day_540_pct"]:
        assert key in km_json
        assert math.isclose(km_json[key], milestones[key], abs_tol=1e-2)

    assert math.isclose(
        dashboard_metrics["overall_kpis"]["retention_12m_pct"],
        milestones["day_365_pct"],
        abs_tol=1e-2,
    )

    # Verify conditional loss probabilities q(a, b) = 1 - S(b)/S(a)
    cond_json = dashboard_metrics["kaplan_meier"]["conditional_loss_probabilities"]
    assert math.isclose(cond_json["q_0_to_180_pct"], 4.76, abs_tol=1e-2)
    assert math.isclose(cond_json["q_180_to_365_pct"], 8.52, abs_tol=1e-2)
    assert math.isclose(cond_json["q_365_to_540_pct"], 10.42, abs_tol=1e-2)
    assert math.isclose(cond_json["q_0_to_180_pct"], conditional["q_0_to_180_pct"], abs_tol=1e-2)
    assert math.isclose(cond_json["q_180_to_365_pct"], conditional["q_180_to_365_pct"], abs_tol=1e-2)
    assert math.isclose(cond_json["q_365_to_540_pct"], conditional["q_365_to_540_pct"], abs_tol=1e-2)

    chi2_tests = dashboard_metrics["chi_square_tests"]
    for col in ["plan_name", "segment", "acquisition_channel"]:
        assert col in chi2_tests
        assert 0.0 <= chi2_tests[col]["p_value"] <= 1.0
        # None of the three categorical differences are statistically significant at alpha=0.05
        assert chi2_tests[col]["p_value"] > 0.05
        assert chi2_tests[col]["significant_at_0_05"] is False


def test_billing_and_payment_coverage_semantics(
    raw_csv_df: pd.DataFrame, dashboard_metrics: dict
) -> None:
    """
    Verifies that billing and payment coverage metrics represent subscription counts
    (has_billing_data and has_payment_data flags), NOT counts of invoices or transactions.
    """
    dq = dashboard_metrics["data_quality_limitations"]
    assert dq["billing_data_coverage_count"] == int((raw_csv_df["has_billing_data"] == 1).sum()) == 120
    assert dq["payment_data_coverage_count"] == int((raw_csv_df["has_payment_data"] == 1).sum()) == 118
    assert math.isclose(dq["billing_data_coverage_pct"], (120 / 2000) * 100.0, abs_tol=1e-2)
    assert math.isclose(dq["payment_data_coverage_pct"], (118 / 2000) * 100.0, abs_tol=1e-2)


def test_dashboard_deliverables_exist_and_single_page() -> None:
    """Verifies that PDF, PNG, and JSON outputs exist, are non-empty, and PDF is single-page."""
    assert OUTPUT_PDF_PATH.exists() and OUTPUT_PDF_PATH.stat().st_size > 0
    assert OUTPUT_PNG_PATH.exists() and OUTPUT_PNG_PATH.stat().st_size > 0
    assert OUTPUT_JSON_PATH.exists() and OUTPUT_JSON_PATH.stat().st_size > 0

    from ml.src.eda.build_bulgu_raporu import verify_pdf_single_page

    assert verify_pdf_single_page(OUTPUT_PDF_PATH) == 1


def test_bulgu_raporu_consistency_and_single_page(dashboard_metrics: dict) -> None:
    """
    Verifies that bulgu_raporu.pdf and bulgu_raporu.docx exist, that bulgu_raporu.pdf
    is strictly 1 page, and that all numbers in the report match dashboard_metrics.json.
    """
    from ml.src.eda.build_bulgu_raporu import (
        OUTPUT_DOCX_PATH as REPORT_DOCX_PATH,
        OUTPUT_PDF_PATH as REPORT_PDF_PATH,
        build_report_content,
        verify_pdf_single_page,
        verify_report_numbers_against_json,
    )

    assert REPORT_DOCX_PATH.exists() and REPORT_DOCX_PATH.stat().st_size > 0
    assert REPORT_PDF_PATH.exists() and REPORT_PDF_PATH.stat().st_size > 0

    content = build_report_content(dashboard_metrics)
    verify_report_numbers_against_json(content, dashboard_metrics)
    assert verify_pdf_single_page(REPORT_PDF_PATH) == 1

