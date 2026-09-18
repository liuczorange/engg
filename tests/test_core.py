import numpy as np
import pandas as pd

from src.features import create_features
from src.split_data import split_buildings


def test_target_and_lags_never_cross_buildings():
    timestamps = pd.date_range("2020-01-01", periods=30, freq="h")
    rows = []
    for building, offset in (("a", 0), ("b", 100)):
        for n, timestamp in enumerate(timestamps):
            rows.append({"building_id": building, "timestamp": timestamp, "electricity": offset + n,
                         "site_id": "s", "primary_use": "Office", "floor_area": 10,
                         "airTemperature": 20})
    out = create_features(pd.DataFrame(rows))
    first_b = out[out.building_id == "b"].iloc[0]
    assert first_b["lag_24"] == 100
    assert first_b["target"] == first_b["current_electricity"] + 1


def test_missing_hour_is_not_treated_as_previous_hour():
    timestamps = pd.date_range("2020-01-01", periods=31, freq="h").delete(25)
    data = pd.DataFrame({"building_id": "a", "timestamp": timestamps,
                         "electricity": np.arange(30.0), "site_id": "s",
                         "primary_use": "Office", "floor_area": 10, "airTemperature": 20})
    out = create_features(data)
    assert pd.Timestamp("2020-01-02 02:00") not in set(out["timestamp"])


def test_building_split_is_disjoint_and_reproducible():
    ids = [f"b{i}" for i in range(20)]
    a = split_buildings(ids, .7, .15, .15, 42)
    b = split_buildings(ids, .7, .15, .15, 42)
    pd.testing.assert_frame_equal(a, b)
    assert a["building_id"].is_unique
    assert set(a["split"]) == {"train", "validation", "test"}
