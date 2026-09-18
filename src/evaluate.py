"""Overall, normalized, macro, and per-building forecast metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_r2(y, prediction) -> float:
    return r2_score(y, prediction) if len(y) >= 2 and np.var(y) > 0 else np.nan


def metric_row(y, prediction) -> dict[str, float]:
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    mae = mean_absolute_error(y, prediction)
    rmse = mean_squared_error(y, prediction) ** 0.5
    mean_demand = np.mean(np.abs(y))
    return {"MAE": mae, "RMSE": rmse, "R2": _safe_r2(y, prediction),
            "NMAE": mae / mean_demand if mean_demand > 0 else np.nan}


def evaluate_predictions(frame: pd.DataFrame, predictions: dict[str, np.ndarray], experiment: str):
    per_building = []
    summary = []
    for model, values in predictions.items():
        evaluated = frame[["building_id", "timestamp", "target"]].copy()
        evaluated["prediction"] = np.asarray(values)
        evaluated = evaluated.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "prediction"])
        building_rows = []
        for building_id, group in evaluated.groupby("building_id"):
            row = {"experiment": experiment, "model": model, "building_id": building_id,
                   "n_observations": len(group), **metric_row(group["target"], group["prediction"])}
            building_rows.append(row)
        per_building.extend(building_rows)
        overall = metric_row(evaluated["target"], evaluated["prediction"])
        per = pd.DataFrame(building_rows)
        overall.update({
            "experiment": experiment, "model": model,
            "macro_MAE": per["MAE"].mean(), "macro_RMSE": per["RMSE"].mean(),
            "n_observations": len(evaluated),
            "n_buildings": evaluated["building_id"].nunique(),
        })
        summary.append(overall)
    return pd.DataFrame(summary), pd.DataFrame(per_building)


def prediction_frame(frame: pd.DataFrame, predictions: dict[str, np.ndarray], experiment: str):
    pieces = []
    for model, values in predictions.items():
        part = frame[["building_id", "timestamp", "target"]].copy()
        part["model"] = model
        part["prediction"] = np.asarray(values)
        part["experiment"] = experiment
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)
