"""Quality reporting, building selection, and safe joining."""
from __future__ import annotations

import numpy as np
import pandas as pd


def data_quality_summary(meter: pd.DataFrame, weather: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meter_stats = meter.groupby("building_id").agg(
        number_of_timestamps=("timestamp", "size"),
        number_of_observations=("electricity", "count"),
        start=("timestamp", "min"), end=("timestamp", "max"),
    )
    meter_stats["expected_hours"] = (
        (meter_stats["end"] - meter_stats["start"]).dt.total_seconds().div(3600).add(1)
    )
    meter_stats["percentage_missing"] = 100 * (
        1 - meter_stats["number_of_observations"] / meter_stats["expected_hours"].clip(lower=1)
    )
    meter_stats["meter_coverage"] = 1 - meter_stats["percentage_missing"] / 100

    weather_keys = weather[["site_id", "timestamp"]].drop_duplicates()
    joined = meter[["building_id", "timestamp"]].merge(metadata[["building_id", "site_id"]], on="building_id")
    joined = joined.merge(weather_keys.assign(_weather=1), on=["site_id", "timestamp"], how="left")
    coverage = joined.groupby("building_id")["_weather"].apply(lambda s: s.notna().mean()).rename("weather_coverage")
    return (meter_stats.join(coverage).reset_index()
            .merge(metadata, on="building_id", how="left")
            .sort_values("building_id"))


def select_buildings(summary: pd.DataFrame, max_buildings: int, min_observations: int,
                     min_meter_coverage: float, min_weather_coverage: float,
                     random_seed: int) -> list[str]:
    valid = summary[
        (summary["number_of_observations"] >= min_observations)
        & (summary["meter_coverage"] >= min_meter_coverage)
        & (summary["weather_coverage"] >= min_weather_coverage)
        & summary["primary_use"].notna()
        & summary["floor_area"].notna()
        & (summary["floor_area"] > 0)
    ].copy()
    if valid.empty:
        raise ValueError("No buildings pass the quality criteria. Inspect data_quality_summary.csv or relax thresholds.")
    if len(valid) > max_buildings:
        valid = valid.sample(max_buildings, random_state=random_seed)
    return sorted(valid["building_id"].tolist())


def join_sources(meter: pd.DataFrame, weather: pd.DataFrame, metadata: pd.DataFrame,
                 buildings: list[str], weather_candidates: list[str],
                 max_weather_missing: float) -> tuple[pd.DataFrame, list[str]]:
    meter = meter[meter["building_id"].isin(buildings)].copy()
    metadata = metadata[metadata["building_id"].isin(buildings)].copy()
    data = meter.merge(metadata, on="building_id", how="inner", validate="many_to_one")
    available = [c for c in weather_candidates if c in weather.columns]
    completeness = weather[available].notna().mean() if available else pd.Series(dtype=float)
    weather_features = completeness[completeness >= (1 - max_weather_missing)].index.tolist()
    if "airTemperature" in available and "airTemperature" not in weather_features:
        weather_features.insert(0, "airTemperature")
    if not weather_features:
        raise ValueError("No sufficiently complete weather variables are available.")
    data = data.merge(
        weather[["site_id", "timestamp", *weather_features]],
        on=["site_id", "timestamp"], how="left", validate="many_to_one",
    )
    data.loc[data["electricity"] < 0, "electricity"] = np.nan
    return data.sort_values(["building_id", "timestamp"]), weather_features
