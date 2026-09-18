"""Linear-regression training."""
from __future__ import annotations

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .modeling import feature_columns, make_preprocessor


def train_linear_regression(train, model_path):
    numeric, categorical = feature_columns(train)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(numeric, categorical, scale=True)),
        ("model", LinearRegression()),
    ])
    # The complete pipeline is fit only on fit rows; validation/test statistics never enter preprocessing.
    pipeline.fit(train[numeric + categorical], train["target"])
    joblib.dump(pipeline, model_path)
    return pipeline, numeric + categorical
