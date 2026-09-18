"""Reproducible building-disjoint and chronological seen-building splits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_buildings(building_ids: list[str], train_ratio: float, val_ratio: float,
                    test_ratio: float, random_seed: int) -> pd.DataFrame:
    if len(building_ids) < 7:
        raise ValueError("At least 7 usable buildings are required for meaningful building-level splits.")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    rng = np.random.default_rng(random_seed)
    ids = np.array(sorted(building_ids), dtype=object)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no training buildings.")
    labels = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    return pd.DataFrame({"building_id": ids, "split": labels})


def make_experiment_sets(features: pd.DataFrame, split: pd.DataFrame, seen_test_ratio: float):
    data = features.merge(split, on="building_id", how="inner", validate="many_to_one")
    train_buildings = data[data["split"] == "train"].copy()
    cutoffs = train_buildings.groupby("building_id")["timestamp"].quantile(1 - seen_test_ratio)
    is_future = train_buildings.apply(lambda r: r["timestamp"] > cutoffs.loc[r["building_id"]], axis=1)
    # Fit rows precede seen-test rows within every training building.
    fit = train_buildings[~is_future].copy()
    seen_test = train_buildings[is_future].copy()
    validation = data[data["split"] == "validation"].copy()
    unseen_test = data[data["split"] == "test"].copy()
    return fit, validation, unseen_test, seen_test
