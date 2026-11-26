"""
Leakage-Safe Preprocessing Pipeline Builders.

Constructs scikit-learn `ColumnTransformer` and `Pipeline` objects so that all
imputation, encoding, and scaling transformers are fit strictly inside each
training fold:
  - Categorical columns: constant imputation with `"Unknown"` + OneHotEncoder
    (`handle_unknown="ignore"`).
  - Numeric columns: median imputation, plus `StandardScaler` only for models
    that require feature scaling (e.g., Logistic Regression).
  - `remainder="drop"` guarantees that non-selected columns never enter a model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

UNKNOWN_CATEGORY_FILL_VALUE = "Unknown"


def build_preprocessor(
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Build a scikit-learn `ColumnTransformer` for categorical and numeric features.

    Parameters
    ----------
    categorical_features:
        Names of categorical columns (including `region` in `region_included`).
    numeric_features:
        Names of numeric columns.
    scale_numeric:
        If True, append `StandardScaler` after median imputation (used for linear
        models such as Logistic Regression). Tree-based models leave this False.
    """
    cat_cols = list(categorical_features)
    num_cols = list(numeric_features)

    if not cat_cols and not num_cols:
        raise ValueError("At least one categorical or numeric feature must be provided.")

    overlap = set(cat_cols) & set(num_cols)
    if overlap:
        raise ValueError(f"Columns cannot be both categorical and numeric: {sorted(overlap)}")

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if cat_cols:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=UNKNOWN_CATEGORY_FILL_VALUE,
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, cat_cols))

    if num_cols:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("numeric", numeric_pipeline, num_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    metadata_columns: Sequence[str] = (),
    target_columns: Sequence[str] = (),
    excluded_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Extract a clean, non-mutating feature matrix `X` from `df` with strict
    leakage and metadata guards.

    Normalizes nullable pandas string missing values (`pd.NA` / `None`) in
    categorical columns to `np.nan` inside an object column on the returned copy
    so scikit-learn `SimpleImputer` handles missingness consistently across
    pandas 2.x and 3.x without modifying the source DataFrame or CSV.
    """
    selected = list(feature_columns)
    forbidden = set(metadata_columns) | set(target_columns) | set(excluded_columns)
    leaked = sorted(set(selected) & forbidden)
    if leaked:
        raise ValueError(
            f"Feature matrix selection attempted to include forbidden columns: {leaked}"
        )

    missing = sorted(set(selected) - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required feature columns: {missing}")

    X = df[selected].copy()

    for col in categorical_columns:
        if col in X.columns:
            # Convert pandas nullable StringDtype / pd.NA / None uniformly to object + np.nan
            X[col] = X[col].astype("object").where(X[col].notna(), np.nan)

    return X
