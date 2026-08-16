from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "temperature_c", "humidity_pct", "is_weekend", "season_code",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h",
]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ts = pd.to_datetime(result["timestamp"])
    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["month"] = ts.dt.month
    result["day_of_year"] = ts.dt.dayofyear
    calculated_weekend = (result["day_of_week"] >= 5).astype(int)
    if "is_weekend" not in result:
        result["is_weekend"] = calculated_weekend
    else:
        result["is_weekend"] = result["is_weekend"].fillna(calculated_weekend)
    calculated_season = pd.Series(
        np.select(
            [result["month"].isin([11, 12, 1, 2]), result["month"].isin([3, 4, 5, 6])],
            [1, 2], default=0,
        ),
        index=result.index,
    )
    if "season_code" not in result:
        result["season_code"] = calculated_season
    else:
        result["season_code"] = result["season_code"].fillna(calculated_season)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    return result


def build_features(frame: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    """Create leakage-safe calendar, lag, and shifted rolling features."""
    result = add_time_features(frame)
    result["lag_1h"] = result["load_kw"].shift(1)
    result["lag_24h"] = result["load_kw"].shift(24)
    result["lag_168h"] = result["load_kw"].shift(168)
    history = result["load_kw"].shift(1)
    result["rolling_mean_24h"] = history.rolling(24).mean()
    result["rolling_std_24h"] = history.rolling(24).std()
    result["rolling_mean_168h"] = history.rolling(168).mean()
    if drop_missing:
        result = result.dropna(subset=DEFAULT_FEATURES + ["load_kw"]).reset_index(drop=True)
    return result


def temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split; random shuffling is invalid for time-series evaluation."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.50.")
    split = int(len(frame) * (1.0 - test_fraction))
    if split < 1 or split >= len(frame):
        raise ValueError("Not enough rows for temporal train/test split.")
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()
