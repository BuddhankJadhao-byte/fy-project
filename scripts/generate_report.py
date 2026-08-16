from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config
from src.data import dataset_summary, load_dataset
from src.reporting import build_forecast_report


def main() -> None:
    config = load_config()
    metrics = pd.read_csv(config["paths"]["output_dir"] / "model_metrics.csv")
    data = load_dataset(config["paths"]["processed_data"])
    forecast_path = config["paths"]["forecast_dir"] / "next_24_hours.csv"
    forecast = pd.read_csv(forecast_path, parse_dates=["timestamp"]) if forecast_path.exists() else None
    output = config["paths"]["report_dir"] / "AI_Load_Forecasting_Report.pdf"
    build_forecast_report(metrics, dataset_summary(data), forecast, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
