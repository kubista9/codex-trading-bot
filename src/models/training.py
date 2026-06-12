from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models.registry import classification_model_registry, regression_model_registry
from src.models.splits import make_temporal_holdout


@dataclass(frozen=True)
class ModelRunResult:
    """Fitted model output for one task/model/horizon."""

    task: str
    horizon: int
    model_name: str
    estimator: Any
    metrics: dict[str, float | int | str]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def train_regression_models(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_col: str,
    horizon: int,
    test_size_days: int = 252,
    min_train_days: int = 252,
    random_state: int = 42,
) -> list[ModelRunResult]:
    """Train baseline regression models for one horizon."""
    return _train_models(
        frame,
        task="regression",
        feature_columns=feature_columns,
        target_col=target_col,
        horizon=horizon,
        models=regression_model_registry(random_state=random_state),
        test_size_days=test_size_days,
        min_train_days=min_train_days,
    )


def train_classification_models(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_col: str,
    horizon: int,
    test_size_days: int = 252,
    min_train_days: int = 252,
    random_state: int = 42,
) -> list[ModelRunResult]:
    """Train baseline classification models for one horizon."""
    return _train_models(
        frame,
        task="classification",
        feature_columns=feature_columns,
        target_col=target_col,
        horizon=horizon,
        models=classification_model_registry(random_state=random_state),
        test_size_days=test_size_days,
        min_train_days=min_train_days,
    )


def _train_models(
    frame: pd.DataFrame,
    *,
    task: str,
    feature_columns: list[str],
    target_col: str,
    horizon: int,
    models: dict[str, Any],
    test_size_days: int,
    min_train_days: int,
) -> list[ModelRunResult]:
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        msg = f"Missing feature columns: {missing_features}"
        raise ValueError(msg)

    split = make_temporal_holdout(
        frame,
        target_col=target_col,
        test_size_days=test_size_days,
        min_train_days=min_train_days,
    )
    train = split.train.dropna(subset=[target_col]).copy()
    test = split.test.dropna(subset=[target_col]).copy()

    if task == "classification" and train[target_col].nunique(dropna=True) < 2:
        return []

    x_train = train[feature_columns]
    x_test = test[feature_columns]
    y_train = train[target_col]
    y_test = test[target_col]

    if task == "classification":
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    results: list[ModelRunResult] = []
    for model_name, estimator in models.items():
        fitted = estimator.fit(x_train, y_train)
        prediction = fitted.predict(x_test)
        metric_values = (
            regression_metrics(y_test, prediction)
            if task == "regression"
            else classification_metrics(y_test, prediction)
        )
        metric_values.update(
            {
                "task": task,
                "horizon": horizon,
                "model_name": model_name,
                "target": target_col,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_start": train["as_of_date"].min().date().isoformat(),
                "train_end": train["as_of_date"].max().date().isoformat(),
                "test_start": test["as_of_date"].min().date().isoformat(),
                "test_end": test["as_of_date"].max().date().isoformat(),
                "cutoff_date": split.cutoff_date.date().isoformat(),
            }
        )

        predictions = _prediction_frame(
            test,
            task=task,
            horizon=horizon,
            model_name=model_name,
            target_col=target_col,
            prediction=prediction,
            estimator=fitted,
            x_test=x_test,
        )
        importance = extract_feature_importance(fitted, feature_columns)
        if not importance.empty:
            importance["task"] = task
            importance["horizon"] = horizon
            importance["model_name"] = model_name

        results.append(
            ModelRunResult(
                task=task,
                horizon=horizon,
                model_name=model_name,
                estimator=fitted,
                metrics=metric_values,
                predictions=predictions,
                feature_importance=importance,
            )
        )

    return results


def _prediction_frame(
    test: pd.DataFrame,
    *,
    task: str,
    horizon: int,
    model_name: str,
    target_col: str,
    prediction: np.ndarray,
    estimator: Any,
    x_test: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["as_of_date", "ticker"] if "ticker" in test.columns else ["as_of_date"]
    output = test[columns].copy()
    output["task"] = task
    output["horizon"] = horizon
    output["model_name"] = model_name
    output["target_col"] = target_col
    output["actual"] = test[target_col].to_numpy()
    output["prediction"] = prediction

    if task == "classification" and hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        if probabilities.shape[1] >= 2:
            output["probability_up"] = probabilities[:, 1]

    return output


def extract_feature_importance(estimator: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract model feature importance or coefficients when available."""
    model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else estimator
    values: np.ndarray | None = None
    importance_type = ""

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        coefficient = np.asarray(model.coef_, dtype=float)
        values = coefficient.reshape(-1)
        if values.size != len(feature_columns):
            values = np.mean(np.abs(coefficient), axis=0)
        importance_type = "coefficient"

    if values is None or values.size != len(feature_columns):
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    output = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "importance_abs": np.abs(values),
            "importance_type": importance_type,
        }
    )
    return output.sort_values("importance_abs", ascending=False).reset_index(drop=True)
