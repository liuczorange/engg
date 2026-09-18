"""Central configuration for the BDG2 forecasting experiment."""
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
# Keep plotting/font caches writable in restricted and CI environments.
_plot_cache = Path(tempfile.gettempdir()) / "bdg2-plot-cache"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_plot_cache / "xdg"))

RANDOM_SEED = 42
MAX_BUILDINGS = 75
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
FORECAST_HORIZON = 1
SEEN_TEST_RATIO = 0.20

# Chosen after inspecting the generated quality table. Override from the CLI.
MIN_METER_COVERAGE = 0.80
MIN_WEATHER_COVERAGE = 0.80
MIN_OBSERVATIONS = 24 * 90
MAX_WEATHER_MISSING = 0.40

WEATHER_CANDIDATES = [
    "airTemperature", "dewTemperature", "seaLvlPressure", "windSpeed",
    "cloudCoverage", "precipDepth1HR", "precipDepth6HR", "windDirection",
]

def ensure_directories() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
