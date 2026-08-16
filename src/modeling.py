from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .features import DEFAULT_FEATURES, temporal_split


@dataclass(frozen=True)
class Metrics:
    mae_kw: float
    rmse_kw: float
    mape_pct: float
    r2: float


def regression_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> Metrics:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    nonzero = np.abs(actual_array) > 1e-9
    mape = np.mean(np.abs((actual_array[nonzero] - predicted_array[nonzero]) / actual_array[nonzero])) * 100
    return Metrics(
        mae_kw=float(mean_absolute_error(actual_array, predicted_array)),
        rmse_kw=float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        mape_pct=float(mape),
        r2=float(r2_score(actual_array, predicted_array)),
    )


def build_random_forest(params: dict[str, Any], random_state: int = 42) -> Pipeline:
    estimator = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 160)),
        max_depth=params.get("max_depth", 18),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", 0.8),
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("regressor", estimator),
    ])


def tune_random_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Pipeline:
    base = Pipeline([
        ("scaler", MinMaxScaler()),
        ("regressor", RandomForestRegressor(n_jobs=-1, random_state=random_state)),
    ])
    search = RandomizedSearchCV(
        base,
        param_distributions={
            "regressor__n_estimators": [40, 60, 80],
            "regressor__max_depth": [8, 10, 12],
            "regressor__min_samples_leaf": [2, 4, 6],
            "regressor__max_features": [0.6, 0.8, 1.0],
        },
        n_iter=10,
        cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_


def arima_210_forecast(history: pd.Series | np.ndarray, steps: int) -> np.ndarray:
    """Fit a transparent ARIMA(2,1,0) baseline by conditional least squares."""
    values = np.asarray(history, dtype=float)
    if len(values) < 4 or steps < 1:
        raise ValueError("ARIMA baseline needs at least four values and one forecast step.")
    differences = np.diff(values)
    X = np.column_stack([differences[1:-1], differences[:-2]])
    y = differences[2:]
    model = LinearRegression().fit(X, y)
    d1, d2 = float(differences[-1]), float(differences[-2])
    level = float(values[-1])
    output = []
    for _ in range(steps):
        next_difference = float(model.predict(np.array([[d1, d2]]))[0])
        level += next_difference
        output.append(level)
        d2, d1 = d1, next_difference
    return np.asarray(output)


def train_and_evaluate(
    featured: pd.DataFrame,
    model_dir: str | Path,
    output_dir: str | Path,
    model_params: dict[str, Any],
    test_fraction: float = 0.2,
    tune: bool = False,
    random_state: int = 42,
    include_arima: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train RF plus fair chronological baselines and persist all artifacts."""
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = temporal_split(featured, test_fraction)
    features = DEFAULT_FEATURES
    X_train, y_train = train[features], train["load_kw"]
    X_test, y_test = test[features], test["load_kw"]

    rf = tune_random_forest(X_train, y_train, random_state) if tune else build_random_forest(model_params, random_state)
    rf.fit(X_train, y_train)
    linear = Pipeline([
        ("scaler", MinMaxScaler()),
        ("regressor", LinearRegression()),
    ]).fit(X_train, y_train)

    predictions = pd.DataFrame({"timestamp": test["timestamp"], "actual_kw": y_test})
    predictions["random_forest_kw"] = rf.predict(X_test)
    predictions["linear_regression_kw"] = linear.predict(X_test)
    predictions["seasonal_naive_kw"] = test["lag_24h"].to_numpy()

    if include_arima:
        predictions["arima_kw"] = arima_210_forecast(y_train, len(test))

    metric_rows = []
    name_map = {
        "random_forest_kw": "Random Forest",
        "linear_regression_kw": "Linear Regression",
        "seasonal_naive_kw": "24-hour Seasonal Naive",
    }
    if include_arima:
        name_map["arima_kw"] = "ARIMA (2,1,0)"
    for column, name in name_map.items():
        metric_rows.append({"model": name, **asdict(regression_metrics(y_test, predictions[column]))})
    metrics = pd.DataFrame(metric_rows).sort_values("mae_kw").reset_index(drop=True)

    importance = pd.DataFrame({
        "feature": features,
        "importance": rf.named_steps["regressor"].feature_importances_,
    }).sort_values(
        "importance", ascending=False
    )
    joblib.dump(rf, model_dir / "random_forest.joblib", compress=9)
    joblib.dump(linear, model_dir / "linear_regression.joblib", compress=9)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    metadata = {
        "features": features,
        "target": "load_kw",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train["timestamp"].min().isoformat(),
        "train_end": train["timestamp"].max().isoformat(),
        "test_start": test["timestamp"].min().isoformat(),
        "test_end": test["timestamp"].max().isoformat(),
        "test_fraction": test_fraction,
        "random_state": random_state,
        "tuned": tune,
        "include_arima": include_arima,
        "preprocessing": {
            "missing_values": "forward-fill (back-fill only for a leading gap)",
            "outliers": "IQR detection followed by removal from signal and forward-fill replacement",
            "continuous_feature_scaling": "MinMaxScaler fitted on training data only",
        },
        "random_forest_params": rf.named_steps["regressor"].get_params(),
        "best_model_by_mae": metrics.iloc[0]["model"],
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return metrics, metadata


def load_model_artifacts(model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "random_forest.joblib")
    with (model_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, metadata
