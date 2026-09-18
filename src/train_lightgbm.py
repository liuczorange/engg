"""Lightweight LightGBM tuning using held-out validation buildings only."""
from __future__ import annotations

import joblib
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
if sys.version_info >= (3, 14):
    # Do not even load its shared library: mixing it with the fallback can abort over OpenMP.
    LGBMRegressor = None
    _LIGHTGBM_ERROR = RuntimeError("LightGBM 4.x is not supported on CPython 3.14")
else:
    try:
        from lightgbm import LGBMRegressor
        _LIGHTGBM_ERROR = None
    except (ImportError, OSError) as exc:  # macOS wheels need libomp; XGBoost is the documented fallback.
        LGBMRegressor = None
        _LIGHTGBM_ERROR = exc

from .modeling import feature_columns, make_preprocessor


PARAMETER_GRID = [
    {"learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "n_estimators": 250, "min_child_samples": 20},
    {"learning_rate": 0.04, "num_leaves": 20, "max_depth": 8, "n_estimators": 350, "min_child_samples": 30},
    {"learning_rate": 0.08, "num_leaves": 15, "max_depth": 6, "n_estimators": 200, "min_child_samples": 40},
]


def train_lightgbm(train, validation, models_dir, random_seed: int):
    numeric, categorical = feature_columns(train)
    columns = numeric + categorical
    best = None
    tuning_rows = []
    for params in PARAMETER_GRID:
        if LGBMRegressor is not None:
            fitted_params = params.copy()
            estimator = LGBMRegressor(
                **fitted_params, random_state=random_seed, n_jobs=-1, verbosity=-1,
                objective="regression_l1",
            )
            backend = "lightgbm"
        else:
            # Import lazily: loading LightGBM and XGBoost together can conflict over OpenMP on macOS.
            from xgboost import XGBRegressor
            fitted_params = {
                "learning_rate": params["learning_rate"],
                "max_depth": max(params["max_depth"], 3),
                "n_estimators": params["n_estimators"],
                "min_child_weight": max(1, params["min_child_samples"] // 10),
            }
            estimator = XGBRegressor(
                **fitted_params,
                random_state=random_seed, n_jobs=-1, objective="reg:absoluteerror",
            )
            backend = "xgboost"
        model = Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical, scale=False)),
            ("model", estimator),
        ])
        model.fit(train[columns], train["target"])
        score = mean_absolute_error(validation["target"], model.predict(validation[columns]))
        tuning_rows.append({**fitted_params, "validation_mae": score})
        if best is None or score < best[0]:
            best = (score, model, fitted_params)
    assert best is not None
    model_path = models_dir / f"{backend}.joblib"
    joblib.dump(best[1], model_path)
    return best[1], columns, tuning_rows, best[2], backend, model_path


def feature_importance(model: Pipeline):
    names = model.named_steps["preprocess"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    return names, values
