from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ensure_output_directories, load_config
from src.data import dataset_summary, load_dataset, prepare_godishala_dataset
from src.features import build_features
from src.forecasting import forecast_future
from src.modeling import load_model_artifacts, train_and_evaluate
from src.reporting import build_forecast_report


def create_figures(config: dict, data: pd.DataFrame) -> None:
    output = config["paths"]["figure_dir"]
    predictions = pd.read_csv(config["paths"]["output_dir"] / "test_predictions.csv", parse_dates=["timestamp"])
    metrics = pd.read_csv(config["paths"]["output_dir"] / "model_metrics.csv")
    importance = pd.read_csv(config["paths"]["output_dir"] / "feature_importance.csv").head(12)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    recent = predictions.tail(24 * 14)
    ax.plot(recent["timestamp"], recent["actual_kw"], label="Actual", linewidth=1.5)
    ax.plot(recent["timestamp"], recent["random_forest_kw"], label="Random Forest", linewidth=1.3)
    ax.set(title="Actual vs Predicted Load - Final 14 Test Days", ylabel="Load (kW)", xlabel="Timestamp")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(metrics["model"], metrics["mae_kw"], color=["#0B6E75", "#4C78A8", "#F58518"])
    ax.set(title="Model MAE Comparison", ylabel="MAE (kW)")
    ax.tick_params(axis="x", rotation=12); fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    view = importance.sort_values("importance")
    ax.barh(view["feature"], view["importance"], color="#0B6E75")
    ax.set(title="Random Forest Feature Importance", xlabel="Importance")
    fig.tight_layout(); fig.savefig(output / "feature_importance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    monthly = data.set_index("timestamp")["load_kw"].resample("ME").mean()
    ax.plot(monthly.index, monthly.values, marker="o", color="#123B5D")
    ax.set(title="Monthly Average Substation Load - 2021", ylabel="Average load (kW)", xlabel="Month")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output / "monthly_load.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete reproducible project pipeline.")
    parser.add_argument("--no-tune", action="store_true", help="Skip time-series RF tuning for a quick development run.")
    args = parser.parse_args()
    config = load_config()
    ensure_output_directories(config)

    data = prepare_godishala_dataset(
        config["paths"]["raw_data"],
        config["paths"]["processed_data"],
        iqr_factor=config["training"]["iqr_factor"],
    )
    featured = build_features(data)
    metrics, metadata = train_and_evaluate(
        featured, config["paths"]["model_dir"], config["paths"]["output_dir"],
        config["training"]["random_forest"], config["training"]["test_fraction"],
        not args.no_tune, config["project"]["random_state"],
        config["training"].get("include_arima", False),
    )
    model, _ = load_model_artifacts(config["paths"]["model_dir"])
    forecast = forecast_future(data, model, 24)
    forecast_path = config["paths"]["forecast_dir"] / "next_24_hours.csv"
    forecast.to_csv(forecast_path, index=False)
    create_figures(config, data)
    report_path = config["paths"]["report_dir"] / "AI_Load_Forecasting_Report.pdf"
    summary = dataset_summary(data)
    build_forecast_report(metrics, summary, forecast, report_path)
    run_summary = {"dataset": summary, "training": metadata, "metrics": metrics.to_dict(orient="records")}
    with (config["paths"]["output_dir"] / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, default=str)
    print("Pipeline completed successfully.")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Dashboard command: streamlit run {ROOT / 'app.py'}")


if __name__ == "__main__":
    main()
