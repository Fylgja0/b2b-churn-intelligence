"""
Unit and Integration Tests for ML Modeling Infrastructure.

Covers:
  1. Model registry algorithm instantiation, exact baseline hyperparameters, and extensibility.
  2. Pipeline construction (StandardScaler only for Logistic Regression; none for tree/dummy).
  3. Scenario A (`region_included`): includes `region`, excludes `region_missing_flag`.
  4. Scenario B (`region_excluded`): excludes both `region` and `region_missing_flag`.
  5. Identical row counts, target distributions, and CV fold indices across both scenarios.
  6. Strict exclusion of metadata IDs and target/leakage columns from model inputs.
  7. Missing `region` values encoded as `"Unknown"` inside the pipeline without mutating input data.
  8. Zero preprocessing leakage from validation fold into training fold imputer/scaler.
  9. Deterministic StratifiedKFold CV splits and reproducible OOF predictions.
 10. Exact-once OOF prediction coverage per observation with all required OOF columns.
 11. End-to-end `run_experiment` on synthetic data in `tmp_path` (never touches real data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.src.models.cross_validation import (
    OOF_REQUIRED_COLUMNS,
    generate_cv_splits,
    run_cross_validation,
    summarize_fold_class_distributions,
)
from ml.src.models.experiment_config import (
    DEFAULT_MANIFEST_PATH,
    PRIMARY_TARGET,
    SCENARIO_REGION_EXCLUDED,
    SCENARIO_REGION_INCLUDED,
    build_experiment_config,
    load_feature_schema,
)
from ml.src.models.model_registry import ModelSpec, get_default_registry
from ml.src.models.preprocessing import (
    UNKNOWN_CATEGORY_FILL_VALUE,
    build_preprocessor,
    prepare_feature_matrix,
)
from ml.src.models.train import run_experiment


def make_synthetic_modeling_df(n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataset matching `feature_manifest.json` schema."""
    rng = np.random.default_rng(seed)

    regions = rng.choice(["APAC", "EMEA", "LATAM", None], size=n_rows, p=[0.25, 0.25, 0.25, 0.25])
    region_missing_flag = [1 if r is None else 0 for r in regions]
    segments = rng.choice(["SMB", "mid_market", "enterprise"], size=n_rows)
    channels = rng.choice(["organic", "paid_search", "partner", "referral", "outbound"], size=n_rows)
    plan_names = rng.choice(["Starter", "Growth", "Scale", "Enterprise"], size=n_rows)
    price_map = {"Starter": 49.0, "Growth": 199.0, "Scale": 799.0, "Enterprise": 1999.0}
    seats_map = {"Starter": 5, "Growth": 20, "Scale": 100, "Enterprise": 250}

    tenure = rng.integers(10, 700, size=n_rows)
    age = tenure + rng.integers(0, 60, size=n_rows)
    start_month = rng.integers(1, 13, size=n_rows)
    start_quarter = ((start_month - 1) // 3) + 1

    # Construct hierarchical churn targets with enough positives for 5-fold CV
    # t90 <= t180 <= t365
    base_score = (
        0.10
        + 0.20 * (plan_names == "Starter").astype(float)
        + 0.15 * (tenure < 150).astype(float)
    )
    u = rng.uniform(0.0, 1.0, size=n_rows)
    t365 = (u < np.clip(base_score + 0.15, 0, 0.8)).astype(int)
    t180 = ((t365 == 1) & (u < np.clip(base_score + 0.03, 0, 0.7))).astype(int)
    t90 = ((t180 == 1) & (u < np.clip(base_score - 0.04, 0, 0.6))).astype(int)

    # Guarantee at least 10 positives in t90 so 5-fold CV always has >=2 per fold
    if t90.sum() < 10:
        idx = np.arange(10)
        t90[idx] = 1
        t180[idx] = 1
        t365[idx] = 1

    return pd.DataFrame(
        {
            "subscription_id": [f"S{i:06d}" for i in range(1, n_rows + 1)],
            "account_id": [f"A{i:06d}" for i in range(1, n_rows + 1)],
            "region": regions,
            "region_missing_flag": region_missing_flag,
            "segment": segments,
            "acquisition_channel": channels,
            "account_age_days_at_t0": age,
            "account_created_at_missing_flag": np.zeros(n_rows, dtype=int),
            "subscription_tenure_days_at_t0": tenure,
            "subscription_start_month": start_month,
            "subscription_start_quarter": start_quarter,
            "plan_name": plan_names,
            "plan_monthly_price": [price_map[p] for p in plan_names],
            "plan_included_seats": [seats_map[p] for p in plan_names],
            "target_churn_90d": t90,
            "target_churn_180d": t180,
            "target_churn_365d": t365,
        }
    )


def test_model_registry_creates_expected_baseline_algorithms() -> None:
    """1. Model registry instantiates all 4 baseline algorithms with exact hyperparameters."""
    registry = get_default_registry()
    assert registry.list_models() == [
        "dummy",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ]

    dummy = registry.create_estimator("dummy", random_state=42)
    assert isinstance(dummy, DummyClassifier)
    assert dummy.strategy == "prior"
    assert registry.get("dummy").is_reference_model is True

    lr = registry.create_estimator("logistic_regression", random_state=42)
    assert isinstance(lr, LogisticRegression)
    assert lr.max_iter == 2000
    assert lr.solver == "lbfgs"
    assert lr.random_state == 42
    assert lr.class_weight is None

    rf = registry.create_estimator("random_forest", random_state=42)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 300
    assert rf.random_state == 42
    assert rf.n_jobs == -1
    assert rf.class_weight is None

    gb = registry.create_estimator("gradient_boosting", random_state=42)
    assert isinstance(gb, GradientBoostingClassifier)
    assert gb.n_estimators == 100
    assert gb.learning_rate == 0.1
    assert gb.max_depth == 3
    assert gb.random_state == 42


def test_model_registry_extensibility() -> None:
    """Registry allows registering new models without altering core code."""
    registry = get_default_registry()
    registry.register(
        ModelSpec(
            name="custom_dummy_uniform",
            display_name="Custom Uniform Dummy",
            estimator_factory=lambda rs: DummyClassifier(strategy="uniform", random_state=rs),
            requires_scaling=False,
        )
    )
    assert "custom_dummy_uniform" in registry.list_models()
    est = registry.create_estimator("custom_dummy_uniform", random_state=7)
    assert isinstance(est, DummyClassifier)
    assert est.strategy == "uniform"


def test_pipelines_built_with_correct_scaling_policy() -> None:
    """2. Pipelines build cleanly; only Logistic Regression scales numeric features."""
    registry = get_default_registry()
    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    _, cat_cols, num_cols = cfg.get_scenario_features()

    for model_name in registry.list_models():
        pipe = registry.create_pipeline(model_name, cat_cols, num_cols, random_state=42)
        assert isinstance(pipe, Pipeline)
        assert list(pipe.named_steps.keys()) == ["preprocessor", "classifier"]

        preprocessor = pipe.named_steps["preprocessor"]
        transformers_by_name = {name: trans for name, trans, _ in preprocessor.transformers}
        num_steps = [step_name for step_name, _ in transformers_by_name["numeric"].steps]

        if model_name == "logistic_regression":
            assert "scaler" in num_steps
        else:
            assert "scaler" not in num_steps


def test_scenario_feature_selection_rules() -> None:
    """3, 4, 5. Scenario A includes `region` and drops `region_missing_flag`; Scenario B drops both."""
    schema = load_feature_schema(DEFAULT_MANIFEST_PATH)
    assert len(schema.all_feature_columns) == 12
    assert len(schema.metadata_columns) == 2
    assert len(schema.target_columns) == 3

    cfg_inc = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel_inc, cat_inc, num_inc = cfg_inc.get_scenario_features()
    assert "region" in sel_inc
    assert "region" in cat_inc
    assert "region_missing_flag" not in sel_inc
    assert "region_missing_flag" not in num_inc
    assert len(sel_inc) == 11

    cfg_exc = build_experiment_config(SCENARIO_REGION_EXCLUDED)
    sel_exc, cat_exc, num_exc = cfg_exc.get_scenario_features()
    assert "region" not in sel_exc
    assert "region" not in cat_exc
    assert "region_missing_flag" not in sel_exc
    assert "region_missing_flag" not in num_exc
    assert len(sel_exc) == 10

    # All non-region features must be identical across scenarios
    assert [c for c in sel_inc if c != "region"] == sel_exc


def test_both_scenarios_preserve_row_count_targets_and_cv_splits() -> None:
    """6, 7. Both scenarios use the exact same rows, targets, and CV fold assignments."""
    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    cfg_inc = build_experiment_config(SCENARIO_REGION_INCLUDED)
    cfg_exc = build_experiment_config(SCENARIO_REGION_EXCLUDED)

    sel_inc, cat_inc, _ = cfg_inc.get_scenario_features()
    sel_exc, cat_exc, _ = cfg_exc.get_scenario_features()

    X_inc = prepare_feature_matrix(
        df,
        sel_inc,
        cat_inc,
        cfg_inc.feature_schema.metadata_columns,
        cfg_inc.feature_schema.target_columns,
        cfg_inc.feature_schema.excluded_columns,
    )
    X_exc = prepare_feature_matrix(
        df,
        sel_exc,
        cat_exc,
        cfg_exc.feature_schema.metadata_columns,
        cfg_exc.feature_schema.target_columns,
        cfg_exc.feature_schema.excluded_columns,
    )

    assert len(X_inc) == len(X_exc) == len(df)

    for target in cfg_inc.targets:
        y_inc = df[target].to_numpy()
        y_exc = df[target].to_numpy()
        assert np.array_equal(y_inc, y_exc)

        splits_inc = generate_cv_splits(y_inc, n_splits=5, shuffle=True, random_state=42)
        splits_exc = generate_cv_splits(y_exc, n_splits=5, shuffle=True, random_state=42)
        for (tr1, va1), (tr2, va2) in zip(splits_inc, splits_exc):
            assert np.array_equal(tr1, tr2)
            assert np.array_equal(va1, va2)


def test_metadata_and_leakage_columns_are_strictly_blocked() -> None:
    """8. Metadata IDs, targets, and leakage columns cannot enter model inputs."""
    df = make_synthetic_modeling_df(n_rows=60, seed=42)
    df["ended_at"] = "2025-05-01"
    df["status"] = "cancelled"
    df["mrr"] = 199.0

    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    schema = cfg.feature_schema
    sel, cat, _ = cfg.get_scenario_features()

    for forbidden_col in [
        "subscription_id",
        "account_id",
        "target_churn_365d",
        "ended_at",
        "status",
        "mrr",
    ]:
        assert forbidden_col not in sel
        with pytest.raises(ValueError, match="forbidden columns"):
            prepare_feature_matrix(
                df=df,
                feature_columns=[*sel, forbidden_col],
                categorical_columns=cat,
                metadata_columns=schema.metadata_columns,
                target_columns=schema.target_columns,
                excluded_columns=schema.excluded_columns,
            )

    # Also verify run_cross_validation rejects metadata columns in X
    registry = get_default_registry()
    _, cat_cols, num_cols = cfg.get_scenario_features()
    pipe = registry.create_pipeline("dummy", cat_cols, num_cols)
    X_bad = df[[*sel, "subscription_id"]].copy()
    with pytest.raises(ValueError, match="Metadata ID columns"):
        run_cross_validation(
            pipeline=pipe,
            X=X_bad,
            y=df[PRIMARY_TARGET],
            metadata_df=df[["subscription_id", "account_id"]],
            scenario=SCENARIO_REGION_INCLUDED,
            target=PRIMARY_TARGET,
            model_name="dummy",
        )


def test_missing_region_handled_as_unknown_without_mutating_source() -> None:
    """9. Missing `region` values become `"Unknown"` inside the pipeline; raw DataFrame stays intact."""
    df = make_synthetic_modeling_df(n_rows=80, seed=11)
    original_null_count = int(df["region"].isna().sum())
    assert original_null_count > 0

    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel, cat, num = cfg.get_scenario_features()
    X = prepare_feature_matrix(
        df,
        sel,
        cat,
        cfg.feature_schema.metadata_columns,
        cfg.feature_schema.target_columns,
        cfg.feature_schema.excluded_columns,
    )

    preprocessor = build_preprocessor(cat, num, scale_numeric=False)
    transformed = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    expected_unknown_col = f"categorical__region_{UNKNOWN_CATEGORY_FILL_VALUE}"
    assert expected_unknown_col in feature_names
    unknown_idx = feature_names.index(expected_unknown_col)
    assert int(transformed[:, unknown_idx].sum()) == original_null_count

    # Source DataFrame must not be mutated
    assert int(df["region"].isna().sum()) == original_null_count


def test_preprocessing_does_not_leak_validation_fold_statistics() -> None:
    """10. Imputer medians and StandardScaler statistics are fit strictly on training fold."""
    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel, cat, num = cfg.get_scenario_features()
    X = prepare_feature_matrix(
        df,
        sel,
        cat,
        cfg.feature_schema.metadata_columns,
        cfg.feature_schema.target_columns,
        cfg.feature_schema.excluded_columns,
    )
    y = df[PRIMARY_TARGET].to_numpy()
    splits = generate_cv_splits(y, n_splits=5, shuffle=True, random_state=42)

    # Inject an extreme outlier and unseen category ONLY into fold 0 validation indices
    fold0_train_idx, fold0_val_idx = splits[0]
    X_perturbed = X.copy()
    X_perturbed.iloc[fold0_val_idx, X_perturbed.columns.get_loc("plan_monthly_price")] = 1_000_000.0
    X_perturbed.iloc[fold0_val_idx, X_perturbed.columns.get_loc("region")] = "ANTARCTICA_UNSEEN"

    registry = get_default_registry()
    pipe = registry.create_pipeline("logistic_regression", cat, num, random_state=42)

    res = run_cross_validation(
        pipeline=pipe,
        X=X_perturbed,
        y=y,
        metadata_df=df[["subscription_id", "account_id"]],
        scenario=SCENARIO_REGION_INCLUDED,
        target=PRIMARY_TARGET,
        model_name="logistic_regression",
        splits=splits,
        retain_fold_pipelines=True,
    )

    fold0_prep = res.fitted_fold_pipelines[0].named_steps["preprocessor"]
    num_pipe = fold0_prep.named_transformers_["numeric"]
    scaler = num_pipe.named_steps["scaler"]
    price_idx = num.index("plan_monthly_price")

    # Scaler mean for fold 0 must equal fold 0 training mean, NOT global mean with 1_000_000
    expected_train_mean = float(X_perturbed.iloc[fold0_train_idx]["plan_monthly_price"].mean())
    assert np.isclose(scaler.mean_[price_idx], expected_train_mean, atol=1e-6)
    assert scaler.mean_[price_idx] < 10_000.0

    # Unseen category in validation fold must NOT appear in fold 0 encoder categories
    cat_pipe = fold0_prep.named_transformers_["categorical"]
    ohe = cat_pipe.named_steps["onehot"]
    region_idx = cat.index("region")
    assert "ANTARCTICA_UNSEEN" not in set(ohe.categories_[region_idx])


def test_cross_validation_is_deterministic_and_covers_every_row_once() -> None:
    """11, 12. CV is deterministic and assigns exactly one OOF prediction per observation."""
    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel, cat, num = cfg.get_scenario_features()
    X = prepare_feature_matrix(
        df,
        sel,
        cat,
        cfg.feature_schema.metadata_columns,
        cfg.feature_schema.target_columns,
        cfg.feature_schema.excluded_columns,
    )
    meta = df[["subscription_id", "account_id"]]
    y = df[PRIMARY_TARGET]

    registry = get_default_registry()
    pipe_a = registry.create_pipeline("logistic_regression", cat, num, random_state=42)
    pipe_b = registry.create_pipeline("logistic_regression", cat, num, random_state=42)

    res_a = run_cross_validation(
        pipeline=pipe_a,
        X=X,
        y=y,
        metadata_df=meta,
        scenario=SCENARIO_REGION_INCLUDED,
        target=PRIMARY_TARGET,
        model_name="logistic_regression",
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    res_b = run_cross_validation(
        pipeline=pipe_b,
        X=X,
        y=y,
        metadata_df=meta,
        scenario=SCENARIO_REGION_INCLUDED,
        target=PRIMARY_TARGET,
        model_name="logistic_regression",
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof_a = res_a.oof_predictions
    oof_b = res_b.oof_predictions

    assert list(oof_a.columns) == list(OOF_REQUIRED_COLUMNS)
    assert len(oof_a) == len(df)
    assert oof_a["subscription_id"].nunique() == len(df)
    assert oof_a["y_probability"].notna().all()
    assert set(oof_a["fold"].unique()) == {0, 1, 2, 3, 4}

    assert np.allclose(oof_a["y_probability"].to_numpy(), oof_b["y_probability"].to_numpy())
    assert np.array_equal(oof_a["fold"].to_numpy(), oof_b["fold"].to_numpy())
    assert np.array_equal(oof_a["y_pred_0_5"].to_numpy(), oof_b["y_pred_0_5"].to_numpy())


def test_fold_class_distribution_reporting() -> None:
    """14. Fold class distributions sum cleanly to full dataset counts."""
    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    y = df["target_churn_90d"].to_numpy()
    splits = generate_cv_splits(y, n_splits=5, shuffle=True, random_state=42)
    fold_dists = summarize_fold_class_distributions(y, splits)

    assert len(fold_dists) == 5
    assert sum(fd["val_n_samples"] for fd in fold_dists) == len(y)
    assert sum(fd["val_n_positive"] for fd in fold_dists) == int(y.sum())
    assert sum(fd["val_n_negative"] for fd in fold_dists) == int((y == 0).sum())


def test_end_to_end_run_experiment_on_synthetic_data(tmp_path: Path) -> None:
    """End-to-end integration test of `run_experiment` in isolated `tmp_path`."""
    from ml.src.models.train import compute_dataset_fingerprint

    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    csv_path = tmp_path / "synthetic_modeling.csv"
    df.to_csv(csv_path, index=False)

    cfg = build_experiment_config(
        scenario=SCENARIO_REGION_INCLUDED,
        targets=[PRIMARY_TARGET],
        models=["dummy", "logistic_regression"],
        dataset_path=csv_path,
        artifacts_base_dir=tmp_path / "artifacts",
        reports_base_dir=tmp_path / "reports",
    )
    summary = run_experiment(cfg, save_artifacts=True, generate_plots=True)

    assert summary["dataset_row_count"] == 100
    assert summary["dataset_fingerprint"] == compute_dataset_fingerprint(csv_path)
    assert len(summary["dataset_fingerprint"]) == 64
    assert len(summary["results"]) == 2
    assert (cfg.reports_dir / "experiment_summary.json").exists()
    assert (cfg.reports_dir / "metrics_summary.csv").exists()
    assert (cfg.reports_dir / "experiment_report.md").exists()
    assert (cfg.reports_dir / f"oof_{PRIMARY_TARGET}_dummy.csv").exists()
    assert (cfg.reports_dir / f"oof_{PRIMARY_TARGET}_logistic_regression.csv").exists()
    assert (cfg.artifacts_dir / f"{PRIMARY_TARGET}_logistic_regression.joblib").exists()
    assert (cfg.reports_dir / "figures" / f"pr_curve_{PRIMARY_TARGET}.png").exists()
    assert (cfg.reports_dir / "figures" / f"roc_curve_{PRIMARY_TARGET}.png").exists()


def test_generate_cv_splits_rejects_insufficient_class_count() -> None:
    """Class count below n_splits raises a descriptive ValueError."""
    # Only 3 positives for 5-fold CV
    y_few_pos = np.array([1, 1, 1] + [0] * 27)
    with pytest.raises(ValueError, match="Insufficient samples per class"):
        generate_cv_splits(y_few_pos, n_splits=5)

    # Only 2 negatives for 5-fold CV
    y_few_neg = np.array([0, 0] + [1] * 28)
    with pytest.raises(ValueError, match="Insufficient samples per class"):
        generate_cv_splits(y_few_neg, n_splits=5)


def test_generate_cv_splits_rejects_non_binary_nan_and_multidim_targets() -> None:
    """Non-binary, fractional, NaN, or 2-D target inputs are strictly rejected."""
    with pytest.raises(ValueError, match="1-dimensional"):
        generate_cv_splits(np.array([[0, 1], [1, 0]]), n_splits=2)

    with pytest.raises(ValueError, match="missing/NaN"):
        generate_cv_splits(np.array([0, 1, np.nan, 0, 1, 0]), n_splits=2)

    with pytest.raises(ValueError, match="fractional or non-binary"):
        generate_cv_splits(np.array([0.0, 0.5, 1.0, 0.0, 1.0, 0.0]), n_splits=2)

    with pytest.raises(ValueError, match="only binary values"):
        generate_cv_splits(np.array([0, 1, 2, 0, 1, 0]), n_splits=2)


def test_run_cross_validation_rejects_invalid_custom_splits() -> None:
    """Overlapping train/val indices, out-of-range indices, or incomplete val coverage fail before fit."""
    df = make_synthetic_modeling_df(n_rows=40, seed=42)
    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel, cat, num = cfg.get_scenario_features()
    X = prepare_feature_matrix(
        df,
        sel,
        cat,
        cfg.feature_schema.metadata_columns,
        cfg.feature_schema.target_columns,
        cfg.feature_schema.excluded_columns,
    )
    meta = df[["subscription_id", "account_id"]]
    y = df[PRIMARY_TARGET]
    registry = get_default_registry()
    pipe = registry.create_pipeline("dummy", cat, num)

    # Overlapping train and validation indices in fold 0
    overlapping_splits = [
        (np.arange(0, 25), np.arange(20, 40)),
        (np.arange(20, 40), np.arange(0, 20)),
    ]
    with pytest.raises(ValueError, match="overlapping train and validation indices"):
        run_cross_validation(
            pipeline=pipe,
            X=X,
            y=y,
            metadata_df=meta,
            scenario=SCENARIO_REGION_INCLUDED,
            target=PRIMARY_TARGET,
            model_name="dummy",
            splits=overlapping_splits,
        )

    # Incomplete validation coverage (missing indices 35..39)
    incomplete_splits = [
        (np.arange(15, 40), np.arange(0, 15)),
        (np.concatenate([np.arange(0, 15), np.arange(35, 40)]), np.arange(15, 35)),
    ]
    with pytest.raises(ValueError, match="appear in validation folds exactly once"):
        run_cross_validation(
            pipeline=pipe,
            X=X,
            y=y,
            metadata_df=meta,
            scenario=SCENARIO_REGION_INCLUDED,
            target=PRIMARY_TARGET,
            model_name="dummy",
            splits=incomplete_splits,
        )

    # Out-of-range index
    out_of_range_splits = [
        (np.arange(0, 20), np.arange(20, 41)),
    ]
    with pytest.raises(ValueError, match="out-of-range"):
        run_cross_validation(
            pipeline=pipe,
            X=X,
            y=y,
            metadata_df=meta,
            scenario=SCENARIO_REGION_INCLUDED,
            target=PRIMARY_TARGET,
            model_name="dummy",
            splits=out_of_range_splits,
        )


def test_run_cross_validation_rejects_misaligned_indices() -> None:
    """Misaligned DataFrame/Series indices between X, metadata_df, and y are rejected."""
    df = make_synthetic_modeling_df(n_rows=40, seed=42)
    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel, cat, num = cfg.get_scenario_features()
    X = prepare_feature_matrix(
        df,
        sel,
        cat,
        cfg.feature_schema.metadata_columns,
        cfg.feature_schema.target_columns,
        cfg.feature_schema.excluded_columns,
    )
    meta_misaligned = df[["subscription_id", "account_id"]].iloc[::-1]
    y = df[PRIMARY_TARGET]
    registry = get_default_registry()
    pipe = registry.create_pipeline("dummy", cat, num)

    with pytest.raises(ValueError, match="Index mismatch between feature matrix X and metadata_df"):
        run_cross_validation(
            pipeline=pipe,
            X=X,
            y=y,
            metadata_df=meta_misaligned,
            scenario=SCENARIO_REGION_INCLUDED,
            target=PRIMARY_TARGET,
            model_name="dummy",
        )


def test_numeric_median_imputation_fit_strictly_on_training_fold() -> None:
    """Numeric SimpleImputer median is fit strictly on the training fold without validation leakage."""
    df = make_synthetic_modeling_df(n_rows=100, seed=42)
    cfg = build_experiment_config(SCENARIO_REGION_INCLUDED)
    sel, cat, num = cfg.get_scenario_features()
    X = prepare_feature_matrix(
        df,
        sel,
        cat,
        cfg.feature_schema.metadata_columns,
        cfg.feature_schema.target_columns,
        cfg.feature_schema.excluded_columns,
    )
    X["account_age_days_at_t0"] = X["account_age_days_at_t0"].astype(float)
    y = df[PRIMARY_TARGET].to_numpy()
    splits = generate_cv_splits(y, n_splits=5, shuffle=True, random_state=42)

    fold0_train_idx, fold0_val_idx = splits[0]
    X_mod = X.copy()
    age_col = X_mod.columns.get_loc("account_age_days_at_t0")

    # Strictly increasing train ages (with one NaN) vs extreme validation ages -> train median != global median
    X_mod.iloc[fold0_train_idx, age_col] = np.arange(1, len(fold0_train_idx) + 1, dtype=float)
    X_mod.iloc[fold0_train_idx[0], age_col] = np.nan
    X_mod.iloc[fold0_val_idx, age_col] = 999_999.0
    expected_train_median = float(np.nanmedian(X_mod.iloc[fold0_train_idx]["account_age_days_at_t0"]))
    global_median = float(np.nanmedian(X_mod["account_age_days_at_t0"]))
    assert expected_train_median != global_median

    registry = get_default_registry()
    pipe = registry.create_pipeline("random_forest", cat, num, random_state=42)

    res = run_cross_validation(
        pipeline=pipe,
        X=X_mod,
        y=y,
        metadata_df=df[["subscription_id", "account_id"]],
        scenario=SCENARIO_REGION_INCLUDED,
        target=PRIMARY_TARGET,
        model_name="random_forest",
        splits=splits,
        retain_fold_pipelines=True,
    )

    fold0_prep = res.fitted_fold_pipelines[0].named_steps["preprocessor"]
    num_imputer = fold0_prep.named_transformers_["numeric"].named_steps["imputer"]
    age_num_idx = num.index("account_age_days_at_t0")
    assert np.isclose(num_imputer.statistics_[age_num_idx], expected_train_median, atol=1e-9)
