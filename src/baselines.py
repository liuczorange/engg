"""Naive one-hour-ahead forecasting baselines."""
import pandas as pd


def baseline_predictions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "persistence": frame["current_electricity"],
        "seasonal_naive": frame["seasonal_naive_prediction"],
    }
