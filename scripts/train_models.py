from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_output_directories, load_config
from src.data import iqr_replace_outliers, load_dataset
from src.features import build_features
from src.modeling import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate load forecasting models.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-tune", action="store_true", help="Skip time-series tuning for a quick development run.")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_output_directories(config)
    frame = iqr_replace_outliers(
        load_dataset(config["paths"]["processed_data"]),
        factor=config["training"]["iqr_factor"],
    )
    frame.to_csv(config["paths"]["processed_data"], index=False, date_format="%Y-%m-%d %H:%M:%S")
    featured = build_features(frame)
    metrics, metadata = train_and_evaluate(
        featured=featured,
        model_dir=config["paths"]["model_dir"],
        output_dir=config["paths"]["output_dir"],
        model_params=config["training"]["random_forest"],
        test_fraction=config["training"]["test_fraction"],
        tune=not args.no_tune,
        random_state=config["project"]["random_state"],
        include_arima=config["training"].get("include_arima", False),
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Best model by MAE: {metadata['best_model_by_mae']}")


if __name__ == "__main__":
    main()
