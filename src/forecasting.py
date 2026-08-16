from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES, build_features


def forecast_future(
    history: pd.DataFrame,
    model: Any,
    horizon: int = 24,
    future_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate an iterative future forecast using known or repeated weather.

    If future weather is unavailable, the most recent 24-hour weather profile is
    repeated. Each predicted load is appended before creating the next row, so
    all lag features remain causally valid.
    """
    if not 1 <= horizon <= 168:
        raise ValueError("Forecast horizon must be between 1 and 168 hours.")
    required = {"timestamp", "load_kw", "temperature_c", "humidity_pct"}
    if not required.issubset(history.columns):
        raise ValueError(f"History is missing: {sorted(required.difference(history.columns))}")
    if len(history) < 168:
        raise ValueError("At least 168 historical hours are required for forecasting.")

    working = history.copy().sort_values("timestamp").reset_index(drop=True)
    start = pd.to_datetime(working["timestamp"].iloc[-1]) + pd.Timedelta(hours=1)
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

    rows: list[dict[str, object]] = []
    for timestamp in future_timestamps:
        candidate = {
            "timestamp": timestamp,
            "load_kw": np.nan,
            "temperature_c": float(weather.loc[timestamp, "temperature_c"]),
            "humidity_pct": float(weather.loc[timestamp, "humidity_pct"]),
        }
        working = pd.concat([working, pd.DataFrame([candidate])], ignore_index=True)
        featured = build_features(working, drop_missing=False)
        row = featured.iloc[[-1]]
        if row[DEFAULT_FEATURES].isna().any().any():
            raise ValueError("Unable to build complete lag features for future forecast.")
        prediction = max(0.0, float(model.predict(row[DEFAULT_FEATURES])[0]))
        working.loc[working.index[-1], "load_kw"] = prediction
        rows.append(
            {
                "timestamp": timestamp,
                "forecast_load_kw": prediction,
                "temperature_c": candidate["temperature_c"],
                "humidity_pct": candidate["humidity_pct"],
            }
        )
    return pd.DataFrame(rows)
