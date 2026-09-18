"""Shared feature selection and train-only preprocessing."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED = {
    "timestamp", "building_id", "site_id", "target", "split",
    "seasonal_naive_prediction", "electricity",
}


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [c for c in frame.columns if c not in EXCLUDED]
    categorical = [c for c in candidates if frame[c].dtype == "object" or str(frame[c].dtype).startswith("category")]
    numeric = [c for c in candidates if c not in categorical and pd.api.types.is_numeric_dtype(frame[c])]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
