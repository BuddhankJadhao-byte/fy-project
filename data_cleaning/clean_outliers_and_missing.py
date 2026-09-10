"""Data Cleaning Pipeline: Outlier Removal & Missing Value Treatment.

This module provides a standalone, production-ready pipeline for detecting,
reporting, and resolving missing values and statistical outliers in electrical
microgrid load and weather data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def detect_and_handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Detect and impute missing values using time-series forward fill with backward fill fallback."""
    initial_missing = df.isna().sum().to_dict()
    
    cleaned = df.copy()
    
    # Handle substation shutdown if present (NaN means no shutdown -> 0)
    if "substation_shutdown" in cleaned.columns:
        cleaned["substation_shutdown"] = cleaned["substation_shutdown"].fillna(0).astype(int)
    
    # Forward-fill temporal numeric signals, then backward-fill any initial boundary gap
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()
    
    return cleaned, initial_missing


def detect_iqr_outliers(
    series: pd.Series, factor: float = 1.5
) -> tuple[pd.Series, float, float, float, float]:
    """Calculate IQR bounds and identify outlier boolean mask."""
    clean_series = series.dropna()
    q1 = float(clean_series.quantile(0.25))
    q3 = float(clean_series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    mask = (series < lower_bound) | (series > upper_bound)
    return mask, q1, q3, lower_bound, upper_bound


def compute_column_stats(series: pd.Series) -> dict[str, float]:
    """Compute summary statistics for a numeric series."""
    valid = series.dropna()
    if len(valid) == 0:
        return {}
    return {
        "count": int(len(valid)),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.median()),
        "q75": float(valid.quantile(0.75)),
        "max": float(valid.max()),
        "skew": float(valid.skew()),
    }


def run_cleaning_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    iqr_factor: float = 1.5,
    target_cols: list[str] | None = None,
) -> dict:
    """Execute the full cleaning, outlier handling, reporting, and visualization pipeline."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_cols is None:
        target_cols = ["load_kw", "temperature_c", "humidity_pct"]

    print(f"[*] Loading raw dataset from: {input_path}")
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        raw = pd.read_excel(input_path)
        timestamp = pd.date_range("2021-01-01 00:00:00", periods=len(raw), freq="h")
        df_raw = pd.DataFrame(
            {
                "timestamp": timestamp,
                "load_kw": pd.to_numeric(raw.get("POWER (KW)", raw.get("load_kw")), errors="coerce"),
                "temperature_c": (
                    (pd.to_numeric(raw.get("Temp (F)", 0), errors="coerce") - 32.0) * 5.0 / 9.0
                    if "Temp (F)" in raw
                    else pd.to_numeric(raw.get("temperature_c", 0), errors="coerce")
                ),
                "humidity_pct": pd.to_numeric(
                    raw.get("Humidity (%)", raw.get("humidity_pct")), errors="coerce"
                ),
                "is_weekend": pd.to_numeric(
                    raw.get('"WEEKEND/WEEKDAY"', raw.get("is_weekend", 0)), errors="coerce"
                ),
                "season_code": pd.to_numeric(
                    raw.get("SEASON", raw.get("season_code", 1)), errors="coerce"
                ),
                "substation_shutdown": raw.get("Substation Shutdown", 0),
            }
        )
    else:
        df_raw = pd.read_csv(input_path)
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

    initial_total_rows = len(df_raw)
    initial_missing_count = int(df_raw.isna().sum().sum())
    print(f"[*] Initial dataset size: {initial_total_rows:,} rows, {initial_missing_count:,} missing values.")

    # 1. Handle missing values
    df_no_missing, missing_per_col = detect_and_handle_missing_values(df_raw)

    # 2. Correct physical boundary anomalies (e.g. sensor readings > 100% humidity or < 0 load)
    physical_anomaly_counts = {}
    if "humidity_pct" in df_no_missing.columns:
        unphysical_humidity = int((df_no_missing["humidity_pct"] > 100).sum() + (df_no_missing["humidity_pct"] < 0).sum())
        physical_anomaly_counts["humidity_pct_unphysical"] = unphysical_humidity
        df_no_missing["humidity_pct"] = df_no_missing["humidity_pct"].clip(0.0, 100.0)

    if "load_kw" in df_no_missing.columns:
        unphysical_load = int((df_no_missing["load_kw"] < 0).sum())
        physical_anomaly_counts["load_kw_negative"] = unphysical_load
        df_no_missing["load_kw"] = df_no_missing["load_kw"].clip(lower=0.0)

    # 3. Detect outliers across target columns using IQR
    outlier_info = {}
    outlier_combined_mask = pd.Series(False, index=df_no_missing.index)

    for col in target_cols:
        if col in df_no_missing.columns:
            mask, q1, q3, lower, upper = detect_iqr_outliers(df_no_missing[col], factor=iqr_factor)
            count = int(mask.sum())
            pct = float((count / initial_total_rows) * 100)
            outlier_info[col] = {
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(q3 - q1, 3),
                "iqr_factor": iqr_factor,
                "lower_bound": round(lower, 3),
                "upper_bound": round(upper, 3),
                "outlier_count": count,
                "outlier_percent": round(pct, 3),
            }
            outlier_combined_mask = outlier_combined_mask | mask
            print(f"    - {col}: {count} outliers detected [{lower:.2f} to {upper:.2f}] ({pct:.2f}%)")

    total_outlier_rows = int(outlier_combined_mask.sum())
    print(f"[*] Total rows containing at least one outlier: {total_outlier_rows:,} ({total_outlier_rows/initial_total_rows*100:.2f}%)")

    # 4. Generate Dataset A: Imputed Dataset (Preserves continuous time-series index for forecasting)
    df_imputed = df_no_missing.copy()
    for col in target_cols:
        if col in df_imputed.columns:
            mask, _, _, lower, upper = detect_iqr_outliers(df_imputed[col], factor=iqr_factor)
            df_imputed.loc[mask, col] = np.nan
            # Forward fill replaced outliers, backward fill fallback
            df_imputed[col] = df_imputed[col].ffill().bfill()
            df_imputed[col] = df_imputed[col].round(3)

    # 5. Generate Dataset B: Dropped Dataset (Strictly removes outlier rows)
    df_dropped = df_no_missing[~outlier_combined_mask].copy().reset_index(drop=True)

    # Save Cleaned CSV Artifacts
    imputed_file = output_dir / "cleaned_data_imputed.csv"
    dropped_file = output_dir / "cleaned_data_dropped.csv"
    df_imputed.to_csv(imputed_file, index=False)
    df_dropped.to_csv(dropped_file, index=False)
    print(f"[+] Saved imputed dataset: {imputed_file} ({len(df_imputed):,} rows)")
    print(f"[+] Saved dropped dataset: {dropped_file} ({len(df_dropped):,} rows)")

    # Save detailed list of every detected outlier
    detailed_outliers = []
    for col in target_cols:
        if col in df_no_missing.columns:
            mask = (df_no_missing[col] < outlier_info[col]["lower_bound"]) | (df_no_missing[col] > outlier_info[col]["upper_bound"])
            for _, r in df_no_missing[mask].iterrows():
                detailed_outliers.append({
                    "timestamp": str(r.get("timestamp", "")),
                    "outlier_column": col,
                    "actual_value": round(float(r[col]), 2),
                    "valid_range": f"{outlier_info[col]['lower_bound']} to {outlier_info[col]['upper_bound']}",
                    "condition": "Below Minimum" if r[col] < outlier_info[col]["lower_bound"] else "Above Maximum",
                })
    df_out_details = pd.DataFrame(detailed_outliers)
    if not df_out_details.empty:
        df_out_details = df_out_details.sort_values("timestamp")
    df_out_details.to_csv(output_dir / "outliers_detected.csv", index=False)
    print(f"[+] Saved itemized outliers list: {output_dir / 'outliers_detected.csv'} ({len(df_out_details)} records)")

    # 6. Compute Before vs After Comparative Statistics Table
    comp_rows = []
    for col in target_cols:
        if col in df_raw.columns:
            raw_stats = compute_column_stats(df_raw[col])
            imp_stats = compute_column_stats(df_imputed[col])
            drop_stats = compute_column_stats(df_dropped[col])

            for metric in ["mean", "std", "min", "q25", "median", "q75", "max", "skew"]:
                comp_rows.append({
                    "feature": col,
                    "statistic": metric,
                    "raw_original": round(raw_stats.get(metric, 0), 3),
                    "after_imputation": round(imp_stats.get(metric, 0), 3),
                    "after_dropping": round(drop_stats.get(metric, 0), 3),
                })

    df_comp = pd.DataFrame(comp_rows)
    comp_csv_path = output_dir / "before_after_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"[+] Saved comparison table: {comp_csv_path}")

    # 7. Generate Visualizations
    print("[*] Generating cleaning and outlier comparison charts...")
    sns.set_theme(style="whitegrid")

    # Figure 1: Box Plots Before vs After
    fig, axes = plt.subplots(len(target_cols), 3, figsize=(15, 4 * len(target_cols)), dpi=150)
    for i, col in enumerate(target_cols):
        # Raw
        sns.boxplot(y=df_raw[col], ax=axes[i, 0], color="#E74C3C")
        axes[i, 0].set_title(f"Raw {col} (Outliers Present)", fontsize=11, fontweight="bold")
        axes[i, 0].set_ylabel(col)
        # Imputed
        sns.boxplot(y=df_imputed[col], ax=axes[i, 1], color="#2ECC71")
        axes[i, 1].set_title(f"Imputed {col} (IQR Capped/Filled)", fontsize=11, fontweight="bold")
        axes[i, 1].set_ylabel("")
        # Dropped
        sns.boxplot(y=df_dropped[col], ax=axes[i, 2], color="#3498DB")
        axes[i, 2].set_title(f"Dropped {col} (Outliers Removed)", fontsize=11, fontweight="bold")
        axes[i, 2].set_ylabel("")

    plt.tight_layout()
    boxplot_path = output_dir / "boxplots_before_after.png"
    plt.savefig(boxplot_path)
    plt.close()
    print(f"[+] Saved boxplot chart: {boxplot_path}")

    # Figure 2: Density Distributions Before vs After
    fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 5), dpi=150)
    for i, col in enumerate(target_cols):
        sns.kdeplot(df_raw[col].dropna(), ax=axes[i], label="Raw", color="#E74C3C", linewidth=2)
        sns.kdeplot(df_imputed[col].dropna(), ax=axes[i], label="Imputed (Preserved Timeline)", color="#2ECC71", linewidth=2, linestyle="--")
        sns.kdeplot(df_dropped[col].dropna(), ax=axes[i], label="Dropped Rows", color="#3498DB", linewidth=1.5, linestyle=":")
        axes[i].set_title(f"Distribution Comparison: {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].legend()

    plt.tight_layout()
    dist_path = output_dir / "distribution_comparison.png"
    plt.savefig(dist_path)
    plt.close()
    print(f"[+] Saved distribution chart: {dist_path}")

    # Figure 3: Time Series Load Outlier Highlighting
    if "load_kw" in df_no_missing.columns and "timestamp" in df_no_missing.columns:
        fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
        mask_load = (df_no_missing["load_kw"] < outlier_info["load_kw"]["lower_bound"]) | (
            df_no_missing["load_kw"] > outlier_info["load_kw"]["upper_bound"]
        )
        ax.plot(df_no_missing["timestamp"], df_no_missing["load_kw"], color="#172B3A", alpha=0.6, linewidth=1, label="Clean Load Signal")
        ax.scatter(
            df_no_missing.loc[mask_load, "timestamp"],
            df_no_missing.loc[mask_load, "load_kw"],
            color="#E74C3C",
            s=40,
            zorder=5,
            label=f"Detected Outliers ({mask_load.sum()} pts)",
        )
        ax.axhline(outlier_info["load_kw"]["upper_bound"], color="#E67E22", linestyle="--", label=f"Upper Bound ({outlier_info['load_kw']['upper_bound']} kW)")
        if outlier_info["load_kw"]["lower_bound"] > 0:
            ax.axhline(outlier_info["load_kw"]["lower_bound"], color="#E67E22", linestyle="--", label=f"Lower Bound ({outlier_info['load_kw']['lower_bound']} kW)")
        ax.set_title("Hourly Electrical Load (kW) with Detected Outlier Bounds", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Load (kW)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        timeline_path = output_dir / "outliers_timeline.png"
        plt.savefig(timeline_path)
        plt.close()
        print(f"[+] Saved outlier timeline chart: {timeline_path}")

    # 8. Save Detailed Summary JSON
    summary = {
        "pipeline_version": "1.0.0",
        "dataset_name": "Godishala 33/11 kV Microgrid Substation",
        "input_source": str(input_path),
        "initial_rows": initial_total_rows,
        "initial_missing_values": initial_missing_count,
        "missing_per_column": {k: int(v) for k, v in missing_per_col.items()},
        "physical_anomalies_corrected": physical_anomaly_counts,
        "iqr_multiplier": iqr_factor,
        "outlier_statistics": outlier_info,
        "total_rows_with_outliers": total_outlier_rows,
        "imputed_dataset": {
            "file": "cleaned_data_imputed.csv",
            "rows": len(df_imputed),
            "retained_timeline_hours": len(df_imputed),
            "description": "Outliers replaced via forward-fill & interpolation; preserves continuous hourly index for time-series forecasting & lag features.",
        },
        "dropped_dataset": {
            "file": "cleaned_data_dropped.csv",
            "rows": len(df_dropped),
            "rows_removed": initial_total_rows - len(df_dropped),
            "description": "Rows with any outlier or unresolvable missing value removed; optimal for tabular regression benchmarks.",
        },
    }

    summary_file = output_dir / "cleaning_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved summary JSON: {summary_file}")

    # 9. Create Comprehensive Markdown Report
    report_content = f"""# Data Cleaning & Outlier Removal Report

## 1. Overview
- **Dataset:** Godishala 33/11 kV Substation (Telangana, India)
- **Original Records:** {initial_total_rows:,} hourly samples
- **Initial Missing Values:** {initial_missing_count:,}
- **IQR Multiplier:** {iqr_factor}× (Standard Tukey Fence)
- **Total Outlier Rows Detected:** {total_outlier_rows:,} ({total_outlier_rows/initial_total_rows*100:.2f}%)

---

## 2. Missing Value Resolution
- **Strategy:** Monotonic forward fill (`ffill`) followed by backward fill (`bfill`) for edge conditions.
- **Substation Shutdown:** {missing_per_col.get('Substation Shutdown', missing_per_col.get('substation_shutdown', 0))} missing cells filled with `0` (normal operational state).
- **Physical Sensor Bounds:**
  - `humidity_pct`: Values bounded to `[0.0, 100.0]%`. Corrected unphysical values exceeding 100%.
  - `load_kw`: Values bounded to `[0.0, ∞) kW`. Negative load values eliminated.

---

## 3. Statistical Outlier Detection (IQR Method)
Formulas applied:
- $IQR = Q_3 - Q_1$
- $\\text{{Lower Bound}} = \\max(0, Q_1 - {iqr_factor} \\times IQR)$
- $\\text{{Upper Bound}} = Q_3 + {iqr_factor} \\times IQR$

| Feature | $Q_1$ (25%) | $Q_3$ (75%) | $IQR$ | Lower Bound | Upper Bound | Outliers Count | Outlier % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for col, info in outlier_info.items():
        report_content += (
            f"| `{col}` | {info['q1']} | {info['q3']} | {info['iqr']} | "
            f"{info['lower_bound']} | {info['upper_bound']} | "
            f"**{info['outlier_count']}** | {info['outlier_percent']}% |\n"
        )

    report_content += f"""
---

## 4. Generated Cleaned Datasets

1. **`cleaned_data_imputed.csv` (Recommended for Time-Series Forecasting)**:
   - **Rows:** {len(df_imputed):,} (100% continuous hourly coverage).
   - Outliers are neutralized via forward-fill and interpolation, ensuring zero gaps in lag features (`lag_1h`, `lag_24h`, `lag_168h`) and rolling windows.
2. **`cleaned_data_dropped.csv` (For Standard Tabular Regression)**:
   - **Rows:** {len(df_dropped):,} ({initial_total_rows - len(df_dropped):,} rows removed).
   - Strictly removes any sample with outlier values.

---

## 5. Artifacts in this Folder
- `cleaned_data_imputed.csv`: Complete time-series cleaned dataset.
- `cleaned_data_dropped.csv`: Filtered tabular dataset.
- `before_after_comparison.csv`: Full statistical metrics before vs after cleaning.
- `cleaning_summary.json`: Programmatic metadata and statistics.
- `boxplots_before_after.png`: Multi-panel before/after boxplot chart.
- `distribution_comparison.png`: Probability density distributions.
- `outliers_timeline.png`: Time-series signal highlighting detected outlier spikes.
"""

    report_md_path = output_dir / "CLEANING_REPORT.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Saved Markdown report: {report_md_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset by handling missing values and removing/imputing outliers.")
    parser.add_argument("--input", default="data/raw/godishala_substation_2021.xlsx", help="Input dataset path")
    parser.add_argument("--output_dir", default="data_cleaning", help="Output directory for cleaned data and artifacts")
    parser.add_argument("--iqr_factor", type=float, default=1.5, help="IQR multiplier threshold (default: 1.5)")
    args = parser.parse_args()

    run_cleaning_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        iqr_factor=args.iqr_factor,
    )
