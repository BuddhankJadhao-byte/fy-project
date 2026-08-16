from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_pipeline import create_figures
from src.config import load_config
from src.data import dataset_summary, load_dataset
from src.forecasting import forecast_future
from src.modeling import load_model_artifacts
from src.reporting import build_forecast_report


def main() -> None:
    config = load_config()
    metrics = pd.read_csv(config["paths"]["output_dir"] / "model_metrics.csv")
    data = load_dataset(config["paths"]["processed_data"])
    model, _ = load_model_artifacts(config["paths"]["model_dir"])
    forecast = forecast_future(data, model, 24)
    forecast_path = config["paths"]["forecast_dir"] / "next_24_hours.csv"
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(forecast_path, index=False)
    create_figures(config, data)
    output = config["paths"]["report_dir"] / "AI_Load_Forecasting_Report.pdf"
    build_forecast_report(metrics, dataset_summary(data), forecast, output)
    print(f"Saved: {forecast_path}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
