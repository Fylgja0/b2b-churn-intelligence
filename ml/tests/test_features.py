"""
Unit Tests for Point-in-Time Safe Feature Engineering & Modeling Dataset.

Verifies:
1. Cohort inclusion rules strictly produce 1,242 active subscriptions at T0 (2024-12-31).
2. Subscription ID and Account ID uniqueness and 1:1 mapping in cohort.
3. Pre-T0 churned and post-T0 started subscriptions are excluded.
4. Target distributions for 90d, 180d, and 365d match verified counts (56, 108, 182).
5. Monotonic hierarchy across churn targets (t90 <= t180 <= t365).
6. Synthetic boundary tests for exact window boundaries and right-censoring.
7. Strict exclusion of target leakage features (ended_at, status, churn, tenure_days).
8. Separation of metadata IDs from feature columns in manifest and dataset.
9. Region missing flag consistency with NULL region values.
10. Plan join integrity and proper renaming without row multiplication.
11. Accurate calculation and non-negativity of account age and subscription tenure.
12. Strict exclusion of unverified fields (seats, mrr) and billing aggregates.
13. Invariance to post-T0 invoice and payment mutations.
14. Deterministic sorting by subscription_id and pipeline reproducibility.
15. Validation errors triggered on invalid negative tenure or age.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from ml.src.features.build_features import (
    FeaturePipeline,
    OBSERVATION_DATE,
    DATASET_CUTOFF_DATE,
    DEFAULT_MODELING_OUTPUT_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_PROCESSED_TABLES_DIR,
)


@pytest.fixture(scope="module")
def feature_pipeline() -> FeaturePipeline:
    return FeaturePipeline()


@pytest.fixture(scope="module")
def pipeline_summary(feature_pipeline: FeaturePipeline) -> dict:
    return feature_pipeline.run()


@pytest.fixture(scope="module")
def modeling_dataset(pipeline_summary: dict) -> pd.DataFrame:
    csv_path = Path(pipeline_summary["output_csv_path"])
    assert csv_path.exists(), f"Modeling CSV not found at {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def feature_manifest() -> dict:
    manifest_path = DEFAULT_REPORTS_DIR / "feature_manifest.json"
    assert manifest_path.exists(), f"Feature manifest not found at {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def clean_tables(feature_pipeline: FeaturePipeline) -> dict[str, pd.DataFrame]:
    return feature_pipeline.load_clean_tables()


def test_cohort_row_count_and_partition_sum(
    modeling_dataset: pd.DataFrame, pipeline_summary: dict, clean_tables: dict[str, pd.DataFrame]
) -> None:
    """Verifies that cohort has exactly 1,242 rows and partition matches 2,000 subscriptions."""
    assert len(modeling_dataset) == 1242
    stats = pipeline_summary["cohort_statistics"]
    assert stats["included_in_cohort"] == 1242
    assert stats["excluded_ended_on_or_before_t0"] == 186
    assert stats["excluded_started_after_t0"] == 571
    assert stats["excluded_missing_started_at"] == 1
    assert stats["total_subscriptions"] == 2000

    subs_df = clean_tables["subscriptions"]
    assert len(subs_df) == 2000


def test_subscription_and_account_id_uniqueness(modeling_dataset: pd.DataFrame) -> None:
    """Verifies that subscription_id and account_id are strictly unique (1:1 grain)."""
    assert modeling_dataset["subscription_id"].nunique() == 1242
    assert modeling_dataset["account_id"].nunique() == 1242
    assert modeling_dataset["subscription_id"].is_unique
    assert modeling_dataset["account_id"].is_unique


def test_no_pre_t0_churned_or_post_t0_started_in_cohort(
    modeling_dataset: pd.DataFrame, clean_tables: dict[str, pd.DataFrame]
) -> None:
    """Verifies that no subscriptions ended <= T0 or started > T0 enter the cohort."""
    subs = clean_tables["subscriptions"].set_index("subscription_id")
    cohort_ids = modeling_dataset["subscription_id"].tolist()

    cohort_subs = subs.loc[cohort_ids]

    # started_at must not be null and must be <= T0
    assert cohort_subs["started_at"].notna().all(), "Cohort must not contain missing started_at"
    assert (cohort_subs["started_at"] <= OBSERVATION_DATE).all(), "Cohort contains subscriptions started after T0!"

    # ended_at must be either null or > T0
    ended = cohort_subs["ended_at"].dropna()
    assert (ended > OBSERVATION_DATE).all(), "Cohort contains subscriptions that ended on or before T0!"


def test_target_distributions_match_expected(modeling_dataset: pd.DataFrame) -> None:
    """Verifies target distributions against expected counts on verified dataset."""
    t90_pos = int(modeling_dataset["target_churn_90d"].sum())
    t90_neg = len(modeling_dataset) - t90_pos
    assert t90_pos == 56
    assert t90_neg == 1186

    t180_pos = int(modeling_dataset["target_churn_180d"].sum())
    t180_neg = len(modeling_dataset) - t180_pos
    assert t180_pos == 108
    assert t180_neg == 1134

    t365_pos = int(modeling_dataset["target_churn_365d"].sum())
    t365_neg = len(modeling_dataset) - t365_pos
    assert t365_pos == 182
    assert t365_neg == 1060


def test_target_hierarchy_and_monotonicity(modeling_dataset: pd.DataFrame) -> None:
    """Verifies logical hierarchy: target_churn_90d <= target_churn_180d <= target_churn_365d."""
    t90 = modeling_dataset["target_churn_90d"]
    t180 = modeling_dataset["target_churn_180d"]
    t365 = modeling_dataset["target_churn_365d"]

    assert (t90 <= t180).all(), "Hierarchy violation: 90d churn must be a subset of 180d churn"
    assert (t180 <= t365).all(), "Hierarchy violation: 180d churn must be a subset of 365d churn"


def test_synthetic_target_window_boundaries_and_inclusion(feature_pipeline: FeaturePipeline) -> None:
    """
    Tests synthetic boundary conditions:
    - Churn on exact window boundary dates must be positive.
    - Churn the day after boundary date must be 0.
    - Subscriptions ended on or before T0 are excluded.
    - NULL ended_at evaluates to 0 across all horizons.
    """
    synthetic_subs = pd.DataFrame([
        # S1: Ends on T0 (excluded from cohort)
        {"subscription_id": "S1", "account_id": "A1", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2024-12-31", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S2: Active, ends 2025-01-01 -> 90d=1, 180d=1, 365d=1
        {"subscription_id": "S2", "account_id": "A2", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2025-01-01", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S3: Ends on 90d boundary 2025-03-31 -> 90d=1, 180d=1, 365d=1
        {"subscription_id": "S3", "account_id": "A3", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2025-03-31", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S4: Ends on 2025-04-01 (day after 90d) -> 90d=0, 180d=1, 365d=1
        {"subscription_id": "S4", "account_id": "A4", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2025-04-01", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S5: Ends on 180d boundary 2025-06-29 -> 90d=0, 180d=1, 365d=1
        {"subscription_id": "S5", "account_id": "A5", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2025-06-29", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S6: Ends on 2025-06-30 (day after 180d) -> 90d=0, 180d=0, 365d=1
        {"subscription_id": "S6", "account_id": "A6", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2025-06-30", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S7: Ends on 365d boundary 2025-12-31 -> 90d=0, 180d=0, 365d=1
        {"subscription_id": "S7", "account_id": "A7", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": "2025-12-31", "status": "cancelled", "seats": 10, "mrr": 199.0},
        # S8: NULL ended_at (active through whole period) -> 90d=0, 180d=0, 365d=0
        {"subscription_id": "S8", "account_id": "A8", "plan_id": "PLN_GROWTH", "started_at": "2024-01-01", "ended_at": None, "status": "active", "seats": 10, "mrr": 199.0},
    ])

    cohort_df, stats = feature_pipeline.create_modeling_cohort(synthetic_subs)
    # S1 should be excluded
    assert len(cohort_df) == 7
    assert "S1" not in cohort_df["subscription_id"].values
    assert stats.excluded_ended_on_or_before_t0 == 1

    horizons = {"90d": "2025-03-31", "180d": "2025-06-29", "365d": "2025-12-31"}
    targets_df, _ = feature_pipeline.compute_target_variables(cohort_df, horizons)
    targets_df = targets_df.set_index("subscription_id")

    # S2: 2025-01-01
    assert targets_df.loc["S2", "target_churn_90d"] == 1
    assert targets_df.loc["S2", "target_churn_180d"] == 1
    assert targets_df.loc["S2", "target_churn_365d"] == 1

    # S3: 2025-03-31 (exact boundary for 90d)
    assert targets_df.loc["S3", "target_churn_90d"] == 1
    assert targets_df.loc["S3", "target_churn_180d"] == 1
    assert targets_df.loc["S3", "target_churn_365d"] == 1

    # S4: 2025-04-01 (1 day after 90d boundary)
    assert targets_df.loc["S4", "target_churn_90d"] == 0
    assert targets_df.loc["S4", "target_churn_180d"] == 1
    assert targets_df.loc["S4", "target_churn_365d"] == 1

    # S5: 2025-06-29 (exact boundary for 180d)
    assert targets_df.loc["S5", "target_churn_90d"] == 0
    assert targets_df.loc["S5", "target_churn_180d"] == 1
    assert targets_df.loc["S5", "target_churn_365d"] == 1

    # S6: 2025-06-30 (1 day after 180d boundary)
    assert targets_df.loc["S6", "target_churn_90d"] == 0
    assert targets_df.loc["S6", "target_churn_180d"] == 0
    assert targets_df.loc["S6", "target_churn_365d"] == 1

    # S7: 2025-12-31 (exact boundary for 365d)
    assert targets_df.loc["S7", "target_churn_90d"] == 0
    assert targets_df.loc["S7", "target_churn_180d"] == 0
    assert targets_df.loc["S7", "target_churn_365d"] == 1

    # S8: NULL ended_at
    assert targets_df.loc["S8", "target_churn_90d"] == 0
    assert targets_df.loc["S8", "target_churn_180d"] == 0
    assert targets_df.loc["S8", "target_churn_365d"] == 0


def test_target_leakage_columns_strictly_excluded(
    modeling_dataset: pd.DataFrame, feature_manifest: dict
) -> None:
    """Verifies that ended_at, status, churn, tenure_days are strictly excluded from dataset and feature space."""
    forbidden = ["ended_at", "status", "churn", "tenure_days"]
    for col in forbidden:
        assert col not in modeling_dataset.columns, f"Target leakage column '{col}' found in modeling dataset!"

    excluded_manifest = {item["name"]: item for item in feature_manifest["excluded_columns"]}
    for col in forbidden:
        assert col in excluded_manifest, f"Forbidden column '{col}' missing from manifest excluded_columns!"
        assert excluded_manifest[col]["category"] == "target_leakage"


def test_metadata_columns_separated_from_features(
    modeling_dataset: pd.DataFrame, feature_manifest: dict
) -> None:
    """Verifies account_id and subscription_id are treated strictly as metadata identifiers."""
    metadata_cols = [m["name"] for m in feature_manifest["metadata_columns"]]
    feature_cols = [f["name"] for f in feature_manifest["feature_columns"]]
    target_cols = [t["name"] for t in feature_manifest["target_columns"]]

    assert "subscription_id" in metadata_cols
    assert "account_id" in metadata_cols
    assert "subscription_id" not in feature_cols
    assert "account_id" not in feature_cols

    # Check complete partition of columns
    assert set(modeling_dataset.columns) == set(metadata_cols + feature_cols + target_cols)
    assert len(modeling_dataset.columns) == 17


def test_region_missing_flag_consistency(modeling_dataset: pd.DataFrame) -> None:
    """Verifies region_missing_flag matches region NULL state across all rows."""
    expected_flag = modeling_dataset["region"].isna().astype(int)
    assert (modeling_dataset["region_missing_flag"] == expected_flag).all()
    # In the verified cohort, missing region count is 321
    assert modeling_dataset["region_missing_flag"].sum() == 321


def test_plan_join_integrity_and_renaming(
    modeling_dataset: pd.DataFrame, clean_tables: dict[str, pd.DataFrame]
) -> None:
    """Verifies plan features are properly renamed, non-null, and did not cause row fanout."""
    assert len(modeling_dataset) == 1242
    assert "plan_monthly_price" in modeling_dataset.columns
    assert "plan_included_seats" in modeling_dataset.columns
    assert "plan_name" in modeling_dataset.columns

    # Original columns should NOT be present under raw names
    assert "monthly_price" not in modeling_dataset.columns
    assert "included_seats" not in modeling_dataset.columns

    # Check values match plans table
    plans_ref = clean_tables["plans"].set_index("plan_name")
    for _, row in modeling_dataset.iterrows():
        p_name = row["plan_name"]
        assert row["plan_monthly_price"] == plans_ref.loc[p_name, "monthly_price"]
        assert row["plan_included_seats"] == plans_ref.loc[p_name, "included_seats"]


def test_age_and_tenure_calculations_and_non_negativity(
    modeling_dataset: pd.DataFrame, clean_tables: dict[str, pd.DataFrame]
) -> None:
    """Verifies account_age_days_at_t0 and subscription_tenure_days_at_t0 accuracy and >= 0."""
    subs = clean_tables["subscriptions"].set_index("subscription_id")
    accs = clean_tables["accounts"].set_index("account_id")
    t0_dt = pd.to_datetime(OBSERVATION_DATE)

    for _, row in modeling_dataset.iterrows():
        sub_id = row["subscription_id"]
        acc_id = row["account_id"]

        started_dt = pd.to_datetime(subs.loc[sub_id, "started_at"])
        created_dt = pd.to_datetime(accs.loc[acc_id, "created_at"])

        expected_tenure = (t0_dt - started_dt).days
        expected_age = (t0_dt - created_dt).days

        assert row["subscription_tenure_days_at_t0"] == expected_tenure
        assert row["account_age_days_at_t0"] == expected_age

    assert (modeling_dataset["subscription_tenure_days_at_t0"] >= 0).all()
    assert (modeling_dataset["account_age_days_at_t0"] >= 0).all()


def test_unverified_and_billing_aggregates_excluded(modeling_dataset: pd.DataFrame) -> None:
    """Verifies that MRR, seats, and billing aggregate columns are strictly excluded."""
    excluded = [
        "mrr",
        "seats",
        "additional_seats",
        "total_invoices",
        "total_payment_attempts",
        "total_payment_failures",
        "payment_failure_rate",
        "total_amount_billed",
        "total_amount_paid",
        "has_billing_data",
        "has_payment_data",
    ]
    for col in excluded:
        assert col not in modeling_dataset.columns, f"Unverified or billing column '{col}' found in dataset!"


def test_post_t0_billing_mutations_do_not_affect_features(
    feature_pipeline: FeaturePipeline, clean_tables: dict[str, pd.DataFrame]
) -> None:
    """Verifies that modifying post-T0 invoices and payments has zero effect on the modeling dataset."""
    tables_copy = {k: v.copy() for k, v in clean_tables.items()}

    # Add artificial post-T0 invoices and payments
    extra_invoices = pd.DataFrame([{
        "invoice_id": "INV_POST_T0_999",
        "subscription_id": "S000001",
        "invoice_date": "2025-05-01",
        "due_date": "2025-05-15",
        "amount_due": 9999.0,
        "invoice_status": "paid",
    }])
    tables_copy["invoices"] = pd.concat([tables_copy["invoices"], extra_invoices], ignore_index=True)

    extra_payments = pd.DataFrame([{
        "payment_id": "PAY_POST_T0_999",
        "invoice_id": "INV_POST_T0_999",
        "attempted_at": "2025-05-02",
        "payment_status": "failed",
        "amount": 9999.0,
    }])
    tables_copy["payments"] = pd.concat([tables_copy["payments"], extra_payments], ignore_index=True)

    # Run cohort and feature extraction with modified tables
    cohort_df, _ = feature_pipeline.create_modeling_cohort(tables_copy["subscriptions"])
    horizons = {"90d": "2025-03-31", "180d": "2025-06-29", "365d": "2025-12-31"}
    cohort_targets, _ = feature_pipeline.compute_target_variables(cohort_df, horizons)
    dataset_mutated = feature_pipeline.engineer_features(
        cohort_targets, tables_copy["accounts"], tables_copy["plans"]
    )

    baseline_path = DEFAULT_MODELING_OUTPUT_DIR / "churn_modeling_dataset.csv"
    baseline_df = pd.read_csv(baseline_path)

    pd.testing.assert_frame_equal(dataset_mutated, baseline_df)


def test_deterministic_ordering_and_reproducibility(
    feature_pipeline: FeaturePipeline, modeling_dataset: pd.DataFrame
) -> None:
    """Verifies that dataset is deterministically sorted by subscription_id and pipeline is reproducible."""
    sub_ids = modeling_dataset["subscription_id"].tolist()
    assert sub_ids == sorted(sub_ids), "Dataset is not deterministically sorted by subscription_id!"

    # Re-run pipeline and compare
    clean_tables = feature_pipeline.load_clean_tables()
    cohort_df, _ = feature_pipeline.create_modeling_cohort(clean_tables["subscriptions"])
    horizons = feature_pipeline.validate_dates_and_censoring(clean_tables)
    cohort_targets, _ = feature_pipeline.compute_target_variables(cohort_df, horizons)
    reproduced_df = feature_pipeline.engineer_features(
        cohort_targets, clean_tables["accounts"], clean_tables["plans"]
    )

    pd.testing.assert_frame_equal(reproduced_df, modeling_dataset)


def test_negative_tenure_or_age_raises_validation_error(feature_pipeline: FeaturePipeline) -> None:
    """Verifies that negative age or tenure triggers a loud ValueError validation failure."""
    invalid_subs = pd.DataFrame([{
        "subscription_id": "S_ERR",
        "account_id": "A_ERR",
        "plan_id": "PLN_GROWTH",
        "started_at": "2025-01-05",  # Started after T0, but forced into engineer_features
        "ended_at": None,
        "target_churn_90d": 0,
        "target_churn_180d": 0,
        "target_churn_365d": 0,
    }])
    accounts = pd.DataFrame([{
        "account_id": "A_ERR",
        "created_at": "2024-01-01",
        "region": "EMEA",
        "segment": "SMB",
        "acquisition_channel": "organic",
    }])
    plans = pd.DataFrame([{
        "plan_id": "PLN_GROWTH",
        "plan_name": "Growth",
        "monthly_price": 199.0,
        "included_seats": 20,
    }])

    with pytest.raises(ValueError, match="Negative subscription tenure detected"):
        feature_pipeline.engineer_features(invalid_subs, accounts, plans)
