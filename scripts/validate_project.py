from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_dataset, validate_dataset
from src.features import DEFAULT_FEATURES, build_features
from src.forecasting import forecast_future
from src.modeling import load_model_artifacts


def main() -> None:
    config = load_config()
    required = [
        ROOT / "app.py", ROOT / "README.md", ROOT / "requirements.txt",
        config["paths"]["processed_data"], config["paths"]["model_dir"] / "random_forest.joblib",
        config["paths"]["output_dir"] / "model_metrics.csv",
        config["paths"]["report_dir"] / "AI_Load_Forecasting_Report.pdf",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing project outputs: {missing}")
    data = load_dataset(config["paths"]["processed_data"])
    validate_dataset(data, 8760)
    featured = build_features(data)
    model, metadata = load_model_artifacts(config["paths"]["model_dir"])
    prediction = model.predict(featured[DEFAULT_FEATURES].tail(48))
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError("Model produced invalid predictions.")
    future = forecast_future(data, model, 24)
    if len(future) != 24 or future.isna().any().any():
        raise AssertionError("Future forecast validation failed.")
    with (config["paths"]["model_dir"] / "metadata.json").open(encoding="utf-8") as handle:
        json.load(handle)
    pdf = config["paths"]["report_dir"] / "AI_Load_Forecasting_Report.pdf"
    if pdf.stat().st_size < 1000:
        raise AssertionError("Generated PDF is unexpectedly small.")
    print(f"PASS: {len(data):,} continuous hourly rows")
    print(f"PASS: {len(DEFAULT_FEATURES)} model features; {metadata['test_rows']:,} test rows")
    print("PASS: model inference, 24-hour forecasting, JSON, CSV, and PDF outputs")


if __name__ == "__main__":
    main()
