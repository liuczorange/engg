# Generalisable short-term building electricity forecasting

This semester-scale machine-learning project predicts whole-building electricity consumption one hour ahead using Building Data Genome Project 2 (BDG2). Its main experiment holds out entire buildings, so every test building is unseen during model training.

## Research question

How accurately can machine-learning models predict one-hour-ahead electricity demand for previously unseen non-residential buildings?

The project compares persistence, 24-hour seasonal-naive, linear regression, and LightGBM forecasts. It also reports a clearly separate future-period experiment on buildings that the models have seen.

## Project layout

```text
data/raw/                 User-supplied BDG2 CSV files
data/processed/           Reserved for optional cached data
outputs/figures/          Five required figures
outputs/metrics/          Quality, split, metrics, tuning, and importance tables
outputs/models/           Saved fitted pipelines
src/                      Loading, features, splitting, models, evaluation, plots
tests/                    Leakage checks and a BDG2-shaped test-data generator
config.py                 Reproducible experiment settings
run_pipeline.py           End-to-end entry point
```

## Dataset and expected files

The implementation was checked against the official BDG2 v1 structure rather than guessed schemas:

- cleaned electricity is a **wide** CSV: `timestamp` followed by one column per `building_id`;
- weather is **long**: `timestamp`, `site_id`, `airTemperature`, `cloudCoverage`, `dewTemperature`, `precipDepth1HR`, `precipDepth6HR`, `seaLvlPressure`, `windDirection`, `windSpeed`;
- metadata contains `building_id`, `site_id`, `primaryspaceusage`, `sqm`, and other optional descriptors.

Download BDG2 from the [official GitHub repository](https://github.com/buds-lab/building-data-genome-project-2) or its linked Zenodo release. Place these three files anywhere below `data/raw/` (nested official repository paths are supported):

```text
electricity_cleaned.csv    # data/meters/cleaned/electricity_cleaned.csv
weather.csv                # data/weather/weather.csv
metadata.csv               # data/metadata/metadata.csv
```

The loader also accepts raw `electricity.csv`. The cleaned electricity file is preferred because BDG2's published cleaning has already removed documented meter anomalies. No other meter type is loaded.

## Installation and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

LightGBM is preferred. On macOS its wheel may require `brew install libomp`; if the native library is unavailable, the code automatically uses the brief's permitted XGBoost fallback and records `tree_model_backend` in `run_manifest.json`.

For a faster first run or a larger later experiment:

```bash
python run_pipeline.py --max-buildings 50
python run_pipeline.py --max-buildings 200
```

Important defaults live in `config.py`, including seed 42, 75 buildings, a 70/15/15 building split, and one-hour horizon. Quality thresholds can be changed at the command line after inspecting `outputs/metrics/data_quality_summary.csv`:

```bash
python run_pipeline.py --min-meter-coverage 0.75 --min-weather-coverage 0.75
```

## Data quality and selection

The pipeline first writes a quality table for every electricity building. Only then does it apply the documented default criteria: at least 90 days of observed readings, 80% meter coverage, 80% matching weather timestamps, present primary use, and positive floor area. The thresholds are conservative starting points, not claims about a universal correct cutoff.

Weather variables with more than 40% missing values are excluded. Outdoor air temperature is retained when present and remaining missing values are handled by train-fitted median imputation. Negative electricity readings are treated as unusable.

## Leakage prevention

- Target, lag, rolling, and seasonal-naive values are constructed separately inside each building.
- Each building is reindexed to a true hourly grid first, so a missing timestamp cannot masquerade as a one-hour lag.
- The target at origin `t` is electricity at `t + 1 hour`.
- Rolling features end at the forecast origin; they never contain the target.
- Train, validation, and test sets contain disjoint building IDs. Individual rows are never randomly mixed.
- All imputers, scaling, and categorical encoding are inside saved scikit-learn pipelines fitted only on training rows.
- LightGBM parameter selection uses validation buildings only. Test buildings are evaluated once after selection.
- The seen-building comparison uses the chronologically final 20% of each training building; model fitting uses only its earlier observations.
- `building_id` and `site_id` are retained only for joining, grouping, splitting, and evaluation. They are excluded from predictors.

## Models and metrics

The persistence prediction is the current reading. The seasonal prediction for target time `t+1` is the reading 24 hours before that target (`t-23`). Linear regression uses scaled numerical inputs and one-hot encoded building use. LightGBM receives the same information and tries three small parameter configurations against held-out validation buildings.

Because consumption cannot be negative, machine-learning predictions are clipped at zero before evaluation. This deterministic post-processing uses no target or future information.

Each experiment reports MAE, RMSE, R², and NMAE (MAE divided by mean absolute demand). It also reports macro averages across per-building errors so large buildings do not dominate the conclusions.

## Generated outputs

```text
outputs/metrics/data_quality_summary.csv
outputs/metrics/building_split.csv
outputs/metrics/model_summary.csv
outputs/metrics/per_building_metrics.csv
outputs/metrics/lightgbm_tuning.csv or xgboost_tuning.csv
outputs/metrics/feature_importance.csv
outputs/metrics/run_manifest.json
outputs/models/linear_regression.joblib
outputs/models/lightgbm.joblib or xgboost.joblib
outputs/figures/actual_vs_predicted.png
outputs/figures/model_mae_comparison.png
outputs/figures/seen_vs_unseen.png
outputs/figures/feature_importance.png
outputs/figures/per_building_error_distribution.png
```

`model_summary.csv` labels rows as `unseen` or `seen`; do not pool the two experiments. The central answer to the research question is the macro error on the `unseen` rows and whether either ML model improves on both naive baselines.

## Tests

```bash
pytest -q
```

Tests check that target/lags never cross building boundaries, gaps are not treated as adjacent hours, and the building split is disjoint and reproducible. `tests/generate_fixture.py` creates a small BDG2-shaped dataset for software validation only; it is not suitable for research conclusions.
