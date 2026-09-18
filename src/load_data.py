"""Locate, validate, and load real BDG2 files."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _find_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path:
    matches: list[Path] = []
    for name in candidates:
        matches.extend(raw_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any of {candidates} below {raw_dir}. "
            "See README.md for the expected BDG2 files."
        )
    return sorted(set(matches), key=lambda p: (len(p.parts), str(p)))[0]


def _pick_column(columns, aliases: tuple[str, ...], label: str) -> str:
    lookup = {str(c).lower().replace("_", ""): c for c in columns}
    for alias in aliases:
        key = alias.lower().replace("_", "")
        if key in lookup:
            return lookup[key]
    raise ValueError(f"Missing {label}; expected one of {aliases}. Found: {list(columns)}")


def inspect_paths(raw_dir: Path) -> dict[str, Path]:
    """Resolve both official repository paths and common flattened downloads."""
    return {
        "meter": _find_file(raw_dir, ("electricity_cleaned.csv", "electricity.csv")),
        "weather": _find_file(raw_dir, ("weather.csv",)),
        "metadata": _find_file(raw_dir, ("metadata.csv",)),
    }


def load_metadata(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    mapping = {
        _pick_column(raw.columns, ("building_id",), "building id"): "building_id",
        _pick_column(raw.columns, ("site_id",), "site id"): "site_id",
        _pick_column(raw.columns, ("primaryspaceusage", "primary_use"), "primary use"): "primary_use",
        _pick_column(raw.columns, ("sqm", "square_meters", "floor_area"), "floor area"): "floor_area",
    }
    raw = raw.rename(columns=mapping)
    # BDG2's optional descriptors are very sparse (some approach 99% missing).
    # Retain only the reliable metadata required by this experiment.
    out = raw[["building_id", "site_id", "primary_use", "floor_area"]].copy()
    out["floor_area"] = pd.to_numeric(out["floor_area"], errors="coerce")
    return out.drop_duplicates("building_id")


def load_weather(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    timestamp = _pick_column(raw.columns, ("timestamp", "datetime"), "weather timestamp")
    site = _pick_column(raw.columns, ("site_id",), "weather site id")
    raw = raw.rename(columns={timestamp: "timestamp", site: "site_id"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    return raw.dropna(subset=["timestamp", "site_id"]).drop_duplicates(["site_id", "timestamp"])


def load_electricity(path: Path, eligible_ids: set[str] | None = None) -> pd.DataFrame:
    """Read official wide-format electricity data and return building/timestamp rows."""
    header = pd.read_csv(path, nrows=0)
    timestamp = _pick_column(header.columns, ("timestamp", "datetime"), "meter timestamp")
    building_columns = [c for c in header.columns if c != timestamp]
    if eligible_ids is not None:
        building_columns = [c for c in building_columns if str(c) in eligible_ids]
    if not building_columns:
        raise ValueError("No electricity columns match buildings with required metadata.")
    wide = pd.read_csv(path, usecols=[timestamp, *building_columns], low_memory=False)
    wide = wide.rename(columns={timestamp: "timestamp"})
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], errors="coerce")
    long = wide.melt(id_vars="timestamp", var_name="building_id", value_name="electricity")
    long["electricity"] = pd.to_numeric(long["electricity"], errors="coerce")
    return long.dropna(subset=["timestamp"]).drop_duplicates(["building_id", "timestamp"])


def load_bdg2(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = inspect_paths(raw_dir)
    metadata = load_metadata(paths["metadata"])
    meter = load_electricity(paths["meter"], set(metadata["building_id"].astype(str)))
    weather = load_weather(paths["weather"])
    return meter, weather, metadata, paths
