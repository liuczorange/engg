#!/usr/bin/env python3
"""Run the complete BDG2 unseen-building forecasting experiment."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.baselines import baseline_predictions
from src.evaluate import evaluate_predictions, prediction_frame
from src.features import create_features
from src.load_data import load_bdg2
from src.plots import actual_vs_predicted, error_distribution, importance_plot, model_comparison, seen_vs_unseen
from src.preprocess import data_quality_summary, join_sources, select_buildings
from src.split_data import make_experiment_sets, split_buildings
from src.train_lightgbm import feature_importance, train_lightgbm
from src.train_linear import train_linear_regression

LOG = logging.getLogger("bdg2")


def _portable_path(path: Path) -> str:
    """Use repository-relative paths in manifests when possible."""
    try:
        return str(Path(path).resolve().relative_to(config.ROOT))
    except ValueError:
        return str(path)


def _predictions(frame, linear, linear_columns, tree_model, tree_columns, tree_name):
    predictions = baseline_predictions(frame)
    # Electricity consumption cannot be negative; clipping is deterministic and uses no future data.
    predictions["linear_regression"] = np.clip(linear.predict(frame[linear_columns]), 0, None)
    predictions[tree_name] = np.clip(tree_model.predict(frame[tree_columns]), 0, None)
    return predictions


def run(raw_dir: Path, max_buildings: int, min_meter_coverage: float,
        min_weather_coverage: float) -> None:
    config.ensure_directories()
    LOG.info("Loading and validating BDG2 files from %s", raw_dir)
    meter, weather, metadata, paths = load_bdg2(raw_dir)
    LOG.info("Files: %s", {k: str(v) for k, v in paths.items()})

    quality = data_quality_summary(meter, weather, metadata)
    quality.to_csv(config.METRICS_DIR / "data_quality_summary.csv", index=False)
    buildings = select_buildings(
        quality, max_buildings=max_buildings, min_observations=config.MIN_OBSERVATIONS,
        min_meter_coverage=min_meter_coverage, min_weather_coverage=min_weather_coverage,
        random_seed=config.RANDOM_SEED,
    )
    LOG.info("Selected %d usable electricity buildings", len(buildings))
    joined, weather_features = join_sources(
        meter, weather, metadata, buildings, config.WEATHER_CANDIDATES, config.MAX_WEATHER_MISSING,
    )
    features = create_features(joined, config.FORECAST_HORIZON)

    split = split_buildings(buildings, config.TRAIN_RATIO, config.VAL_RATIO,
                            config.TEST_RATIO, config.RANDOM_SEED)
    split.to_csv(config.METRICS_DIR / "building_split.csv", index=False)
    fit, validation, unseen_test, seen_test = make_experiment_sets(features, split, config.SEEN_TEST_RATIO)
    if min(map(len, (fit, validation, unseen_test, seen_test))) == 0:
        raise ValueError("One experiment partition is empty after feature creation.")

    LOG.info("Training linear regression on %d rows", len(fit))
    linear, linear_columns = train_linear_regression(fit, config.MODELS_DIR / "linear_regression.joblib")
    LOG.info("Tuning the preferred tree model on validation buildings")
    lightgbm, lightgbm_columns, tuning, best_params, tree_backend, tree_model_path = train_lightgbm(
        fit, validation, config.MODELS_DIR, config.RANDOM_SEED,
    )
    pd.DataFrame(tuning).to_csv(config.METRICS_DIR / f"{tree_backend}_tuning.csv", index=False)

    summaries, per_building, prediction_frames = [], [], []
    for experiment, frame in (("unseen", unseen_test), ("seen", seen_test)):
        predictions = _predictions(frame, linear, linear_columns, lightgbm, lightgbm_columns, tree_backend)
        summary, per = evaluate_predictions(frame, predictions, experiment)
        summaries.append(summary); per_building.append(per)
        prediction_frames.append(prediction_frame(frame, predictions, experiment))
    summary = pd.concat(summaries, ignore_index=True)
    per = pd.concat(per_building, ignore_index=True)
    prediction_data = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(config.METRICS_DIR / "model_summary.csv", index=False)
    per.to_csv(config.METRICS_DIR / "per_building_metrics.csv", index=False)

    names, values = feature_importance(lightgbm)
    clean_names = [name.replace("numeric__", "").replace("categorical__", "") for name in names]
    importance = pd.DataFrame({"feature": clean_names, "importance": values}).sort_values("importance", ascending=False)
    importance.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)
    importance_plot(importance, config.FIGURES_DIR / "feature_importance.png")
    actual_vs_predicted(prediction_data, config.FIGURES_DIR / "actual_vs_predicted.png")
    model_comparison(summary, config.FIGURES_DIR / "model_mae_comparison.png")
    seen_vs_unseen(summary, config.FIGURES_DIR / "seen_vs_unseen.png")
    error_distribution(per, config.FIGURES_DIR / "per_building_error_distribution.png")

    manifest = {
        "selected_buildings": len(buildings), "weather_features": weather_features,
        "feature_columns": linear_columns, "best_tree_parameters": best_params,
        "random_seed": config.RANDOM_SEED, "tree_model_backend": tree_backend,
        "tree_model_path": _portable_path(tree_model_path),
        "source_files": {key: _portable_path(value) for key, value in paths.items()},
    }
    (config.METRICS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("Complete. Unseen-building results:\n%s", summary[summary.experiment == "unseen"].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--max-buildings", type=int, default=config.MAX_BUILDINGS)
    parser.add_argument("--min-meter-coverage", type=float, default=config.MIN_METER_COVERAGE)
    parser.add_argument("--min-weather-coverage", type=float, default=config.MIN_WEATHER_COVERAGE)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run(args.raw_data_dir, args.max_buildings, args.min_meter_coverage, args.min_weather_coverage)
