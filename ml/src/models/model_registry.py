"""
Extensible Baseline Model Registry.

Registers the initial baseline algorithms with fixed, reproducible starter
hyperparameters (no SMOTE, no class weighting, no hyperparameter tuning):
  1. `dummy`: DummyClassifier(strategy="prior") [reference model]
  2. `logistic_regression`: LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
  3. `random_forest`: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
  4. `gradient_boosting`: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
     max_depth=3, random_state=42)

New algorithms can be registered via `ModelRegistry.register(...)` without
changing the training or evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.src.models.experiment_config import RANDOM_STATE
from ml.src.models.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification for a registered classification model."""

    name: str
    display_name: str
    estimator_factory: Callable[[int], BaseEstimator]
    requires_scaling: bool
    is_reference_model: bool = False
    description: str = ""


class ModelRegistry:
    """Registry that constructs estimators and full preprocessing+model pipelines."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec, overwrite: bool = False) -> None:
        if not spec.name or not spec.name.strip():
            raise ValueError("ModelSpec.name must be a non-empty string.")
        if spec.name in self._specs and not overwrite:
            raise ValueError(f"Model '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(
                f"Unknown model '{name}'. Available models: {self.list_models()}"
            )
        return self._specs[name]

    def list_models(self) -> list[str]:
        return list(self._specs.keys())

    def create_estimator(self, name: str, random_state: int = RANDOM_STATE) -> BaseEstimator:
        spec = self.get(name)
        return spec.estimator_factory(random_state)

    def create_pipeline(
        self,
        name: str,
        categorical_features: Sequence[str],
        numeric_features: Sequence[str],
        random_state: int = RANDOM_STATE,
    ) -> Pipeline:
        """
        Build a complete scikit-learn `Pipeline` combining scenario preprocessing
        and the registered classifier.
        """
        spec = self.get(name)
        preprocessor = build_preprocessor(
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            scale_numeric=spec.requires_scaling,
        )
        estimator = spec.estimator_factory(random_state)
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )


def _build_dummy(_random_state: int) -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def _build_logistic_regression(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )


def _build_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )


def _build_gradient_boosting(random_state: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
    )


def get_default_registry() -> ModelRegistry:
    """Return a `ModelRegistry` pre-populated with the 4 baseline models."""
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            name="dummy",
            display_name="Dummy Classifier (Prior)",
            estimator_factory=_build_dummy,
            requires_scaling=False,
            is_reference_model=True,
            description="Reference prior-probability baseline; not a candidate model.",
        )
    )
    registry.register(
        ModelSpec(
            name="logistic_regression",
            display_name="Logistic Regression",
            estimator_factory=_build_logistic_regression,
            requires_scaling=True,
            is_reference_model=False,
            description="Linear baseline with StandardScaler on numeric features and default class weights.",
        )
    )
    registry.register(
        ModelSpec(
            name="random_forest",
            display_name="Random Forest Classifier",
            estimator_factory=_build_random_forest,
            requires_scaling=False,
            is_reference_model=False,
            description="Ensemble of 300 decision trees without feature scaling.",
        )
    )
    registry.register(
        ModelSpec(
            name="gradient_boosting",
            display_name="Gradient Boosting Classifier",
            estimator_factory=_build_gradient_boosting,
            requires_scaling=False,
            is_reference_model=False,
            description="Gradient boosted trees (n_estimators=100, lr=0.1, max_depth=3).",
        )
    )
    return registry
