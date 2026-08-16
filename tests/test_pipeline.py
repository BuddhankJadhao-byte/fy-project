from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data import validate_dataset
from src.features import DEFAULT_FEATURES, build_features, temporal_split
from src.forecasting import forecast_future
from src.modeling import regression_metrics
from src.reporting import build_forecast_report


def sample_data(hours: int = 24 * 40) -> pd.DataFrame:
    timestamp = pd.date_range("2021-01-01", periods=hours, freq="h")
    hour = timestamp.hour.to_numpy()
    load = 1500 + 400 * np.sin(2 * np.pi * hour / 24) + np.arange(hours) * 0.05
    return pd.DataFrame({
        "timestamp": timestamp,
        "load_kw": load,
        "temperature_c": 28 + 5 * np.sin(2 * np.pi * hour / 24),
        "humidity_pct": 65 - 10 * np.sin(2 * np.pi * hour / 24),
    })


class PipelineTests(unittest.TestCase):
    def test_validation_and_features(self) -> None:
        data = sample_data()
        validate_dataset(data)
        featured = build_features(data)
        self.assertEqual(len(featured), len(data) - 168)
        self.assertFalse(featured[DEFAULT_FEATURES].isna().any().any())
        self.assertAlmostEqual(featured.iloc[0]["lag_168h"], data.iloc[0]["load_kw"])

    def test_temporal_split_is_ordered(self) -> None:
        train, test = temporal_split(build_features(sample_data()), 0.2)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())

    def test_metrics(self) -> None:
        metrics = regression_metrics([100, 200, 300], [110, 190, 310])
        self.assertAlmostEqual(metrics.mae_kw, 10.0)
        self.assertGreater(metrics.r2, 0.9)

    def test_iterative_forecast(self) -> None:
        data = sample_data()
        featured = build_features(data)
        model = RandomForestRegressor(n_estimators=10, random_state=1).fit(featured[DEFAULT_FEATURES], featured["load_kw"])
        forecast = forecast_future(data, model, 24)
        self.assertEqual(len(forecast), 24)
        self.assertTrue(np.isfinite(forecast["forecast_load_kw"]).all())
        self.assertTrue((forecast["forecast_load_kw"] >= 0).all())

    def test_pdf_report(self) -> None:
        metrics = pd.DataFrame([{"model": "Random Forest", "mae_kw": 10, "rmse_kw": 12, "mape_pct": 1, "r2": 0.98}])
        pdf = build_forecast_report(metrics, {"rows": 8760, "start": "2021-01-01"})
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 1000)


if __name__ == "__main__":
    unittest.main()
