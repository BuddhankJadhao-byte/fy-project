from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES


def forecast_future(
    history: pd.DataFrame,
    model: Any,
    horizon: int = 24,
    future_weather: pd.DataFrame | None = None,
    start_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate an iterative forecast beginning at any date after the history.

    If future weather is unavailable, the most recent 24-hour weather profile is
    repeated. For a start date beyond the next historical hour, the latest 168
    observed loads anchor the scenario; predictions then proceed recursively.
    """
    if not 1 <= horizon <= 24 * 366:
        raise ValueError("Forecast horizon must be between 1 hour and 366 days.")
    required = {"timestamp", "load_kw", "temperature_c", "humidity_pct"}
    if not required.issubset(history.columns):
        raise ValueError(f"History is missing: {sorted(required.difference(history.columns))}")
    if len(history) < 168:
        raise ValueError("At least 168 historical hours are required for forecasting.")

    working = history.copy().sort_values("timestamp").reset_index(drop=True)
    next_hour = pd.Timestamp(working["timestamp"].iloc[-1]) + pd.Timedelta(hours=1)
    start = pd.Timestamp(start_timestamp) if start_timestamp is not None else next_hour
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    start = start.floor("h")
    if start < next_hour:
        raise ValueError(f"Forecast start must be on or after {next_hour}.")
    future_timestamps = pd.date_range(start, periods=horizon, freq="h")

    if future_weather is not None:
        weather = future_weather.copy()
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])
        weather = weather.set_index("timestamp").reindex(future_timestamps)
        if weather[["temperature_c", "humidity_pct"]].isna().any().any():
            raise ValueError("Future weather must cover every forecast timestamp.")
    else:
        profile = working.tail(24)[["temperature_c", "humidity_pct"]].reset_index(drop=True)
        weather = pd.concat([profile] * int(np.ceil(horizon / 24)), ignore_index=True).iloc[:horizon]
        weather.index = future_timestamps

    load_history = working["load_kw"].astype(float).tolist()
    rows: list[dict[str, object]] = []
    for timestamp in future_timestamps:
        month = timestamp.month
        recent_24 = np.asarray(load_history[-24:], dtype=float)
        recent_168 = np.asarray(load_history[-168:], dtype=float)
        features = {
            "temperature_c": float(weather.loc[timestamp, "temperature_c"]),
            "humidity_pct": float(weather.loc[timestamp, "humidity_pct"]),
            "is_weekend": int(timestamp.dayofweek >= 5),
            "season_code": 1 if month in (11, 12, 1, 2) else 2 if month in (3, 4, 5, 6) else 0,
            "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
            "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
            "dow_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "lag_1h": load_history[-1],
            "lag_24h": load_history[-24],
            "lag_168h": load_history[-168],
            "rolling_mean_24h": recent_24.mean(),
            "rolling_std_24h": recent_24.std(ddof=1),
            "rolling_mean_168h": recent_168.mean(),
        }
        row = pd.DataFrame([[features[name] for name in DEFAULT_FEATURES]], columns=DEFAULT_FEATURES)
        prediction = max(0.0, float(model.predict(row)[0]))
        load_history.append(prediction)
        rows.append(
            {
                "timestamp": timestamp,
                "forecast_load_kw": prediction,
                "temperature_c": features["temperature_c"],
                "humidity_pct": features["humidity_pct"],
            }
        )
    return pd.DataFrame(rows)
