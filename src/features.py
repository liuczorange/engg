"""Leakage-safe, timestamp-aware feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hourly_group(group: pd.DataFrame) -> pd.DataFrame:
    building = group["building_id"].iloc[0]
    group = group.sort_values("timestamp").set_index("timestamp")
    full_index = pd.date_range(group.index.min(), group.index.max(), freq="h")
    group = group.reindex(full_index)
    group.index.name = "timestamp"
    group["building_id"] = building
    # Metadata is constant; weather gaps are left for train-fitted imputers.
    for col in ["site_id", "primary_use", "floor_area"]:
        if col in group:
            group[col] = group[col].ffill().bfill()
    return group.reset_index()


def create_features(data: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
    frames = []
    for _, group in data.groupby("building_id", sort=False):
        g = _hourly_group(group)
        y = g["electricity"]
        # Every shift stays inside one building. Reindexing first makes a lag mean an actual hour.
        g["target"] = y.shift(-forecast_horizon)
        g["current_electricity"] = y
        for lag in (1, 2, 3, 24):
            g[f"lag_{lag}"] = y.shift(lag)
        # At forecast origin t, current and older readings are known; no target/future enters a window.
        for window in (3, 6, 24):
            g[f"rolling_mean_{window}h"] = y.rolling(window, min_periods=window).mean()
        g["rolling_std_24h"] = y.rolling(24, min_periods=24).std()
        # Target at t+1 compared with the observation at t+1-24 (23 rows behind origin t).
        g["seasonal_naive_prediction"] = y.shift(24 - forecast_horizon)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    ts = out["timestamp"]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out.dropna(subset=["target", "current_electricity", "lag_1", "lag_2", "lag_3", "lag_24",
                               "rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h",
                               "rolling_std_24h", "seasonal_naive_prediction"]).reset_index(drop=True)
