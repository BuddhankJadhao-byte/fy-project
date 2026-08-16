from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_output_directories, load_config
from src.data import dataset_summary, prepare_godishala_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Godishala hourly load dataset.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_output_directories(config)
    frame = prepare_godishala_dataset(config["paths"]["raw_data"], config["paths"]["processed_data"])
    summary = dataset_summary(frame)
    print(f"Prepared {summary['rows']:,} hourly rows: {summary['start']} to {summary['end']}")
    print(f"Saved: {config['paths']['processed_data']}")


if __name__ == "__main__":
    main()
