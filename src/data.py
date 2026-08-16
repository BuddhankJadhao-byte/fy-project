from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "load_kw", "temperature_c", "humidity_pct")


def prepare_godishala_dataset(
    source_path: str | Path,
    output_path: str | Path,
    iqr_factor: float = 3.0,
) -> pd.DataFrame:
    """Convert the published Godishala workbook into a clean forecasting CSV.

    The workbook contains one row per hour and 8,760 rows. Its DATE cells are
    populated only once per day and TIME is stored as hour-ending text. A
    monotonic hourly index is therefore generated from the documented 2021
    coverage. Voltage/current/power-factor fields are deliberately excluded to
    avoid target leakage because they directly determine three-phase power.
    """
    source_path, output_path = Path(source_path), Path(output_path)
    raw = pd.read_excel(source_path)
    if len(raw) != 8760:
        raise ValueError(f"Expected 8,760 hourly rows, found {len(raw):,}.")

    needed = {
        "POWER (KW)", '"WEEKEND/WEEKDAY"', "SEASON", "Temp (F)",
        "Humidity (%)", "Substation Shutdown",
    }
    missing = needed.difference(raw.columns)
    if missing:
        raise ValueError(f"Source workbook is missing columns: {sorted(missing)}")

    timestamp = pd.date_range("2021-01-01 00:00:00", periods=len(raw), freq="h")
    clean = pd.DataFrame(
        {
            "timestamp": timestamp,
            "load_kw": pd.to_numeric(raw["POWER (KW)"], errors="coerce"),
            "temperature_c": (pd.to_numeric(raw["Temp (F)"], errors="coerce") - 32.0) * 5.0 / 9.0,
            "humidity_pct": pd.to_numeric(raw["Humidity (%)"], errors="coerce"),
            "is_weekend": pd.to_numeric(raw['"WEEKEND/WEEKDAY"'], errors="coerce"),
            "season_code": pd.to_numeric(raw["SEASON"], errors="coerce"),
            "substation_shutdown": raw["Substation Shutdown"].fillna(0).astype(int),
        }
    )
    numeric = clean.columns.drop("timestamp")
    clean[numeric] = forward_fill_missing(clean[numeric])
    # Six published humidity readings are 101-102%; cap them at the physical limit.
    clean["humidity_pct"] = clean["humidity_pct"].clip(0, 100)
    clean["temperature_c"] = clean["temperature_c"].round(3)
    clean["load_kw"] = clean["load_kw"].round(3)
    clean = iqr_replace_outliers(clean, factor=iqr_factor)

    validate_dataset(clean, minimum_hours=8760)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return clean


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a project-compatible CSV."""
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError("CSV must contain a 'timestamp' column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in frame.columns.drop("timestamp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame, minimum_hours: int = 24 * 30) -> None:
    """Validate schema, types, ordering, continuity, and numeric ranges."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) < minimum_hours:
        raise ValueError(f"At least {minimum_hours:,} hourly rows are required.")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing or invalid values.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be in increasing order.")
    intervals = frame["timestamp"].diff().dropna()
    if not intervals.eq(pd.Timedelta(hours=1)).all():
        bad = int((~intervals.eq(pd.Timedelta(hours=1))).sum())
        raise ValueError(f"Dataset is not continuous hourly data ({bad} irregular intervals).")
    if (frame["load_kw"] < 0).any():
        raise ValueError("Load values cannot be negative.")
    if not frame["humidity_pct"].between(0, 100).all():
        raise ValueError("Humidity must be within 0-100 percent.")


def dataset_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return serializable dataset quality and range statistics."""
    return {
        "rows": int(len(frame)),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "hours": int((frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 3600 + 1),
        "missing_values": int(frame.isna().sum().sum()),
        "load_mean_kw": float(frame["load_kw"].mean()),
        "load_min_kw": float(frame["load_kw"].min()),
        "load_max_kw": float(frame["load_kw"].max()),
        "temperature_mean_c": float(frame["temperature_c"].mean()),
        "humidity_mean_pct": float(frame["humidity_pct"].mean()),
        "shutdown_hours": int(frame.get("substation_shutdown", pd.Series(dtype=int)).sum()),
        "iqr_outliers_replaced": int(frame.get("load_outlier_flag", pd.Series(dtype=int)).sum()),
    }


def forward_fill_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the synopsis-specified forward fill, with a safe leading-value fallback."""
    return frame.ffill().bfill()


def iqr_replace_outliers(
    frame: pd.DataFrame,
    column: str = "load_kw",
    factor: float = 3.0,
) -> pd.DataFrame:
    """Detect IQR outliers and replace them without breaking the hourly timeline.

    Dropping complete rows would create missing timestamps and invalidate lag features.
    Values outside the IQR bounds are therefore removed from the signal, marked, and
    forward-filled as required by the submitted synopsis.
    """
    result = frame.copy()
    q1, q3 = result[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = max(0.0, q1 - factor * iqr), q3 + factor * iqr
    mask = ~result[column].between(lower, upper)
    result["load_outlier_flag"] = mask.astype(int)
    result.loc[mask, column] = np.nan
    result[column] = forward_fill_missing(result[[column]])[column]
    return result


def iqr_clip(frame: pd.DataFrame, column: str = "load_kw", factor: float = 3.0) -> pd.DataFrame:
    """Backward-compatible alias for the synopsis-aligned IQR replacement step."""
    return iqr_replace_outliers(frame, column=column, factor=factor)
