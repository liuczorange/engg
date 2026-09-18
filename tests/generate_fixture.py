"""Generate a small BDG2-shaped dataset for tests only (not research results)."""
from pathlib import Path
import csv
import math
import random


def generate(root: Path, buildings=10, days=100, seed=7):
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ids = [f"Site{n % 3}_education_Building{n:02d}" for n in range(buildings)]
    sites = [f"Site{n}" for n in range(3)]
    with (root / "metadata.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["building_id", "site_id", "primaryspaceusage", "sqm", "electricity"])
        for n, building in enumerate(ids):
            writer.writerow([building, sites[n % 3], "Education" if n % 2 else "Office", 1000 + 350 * n, "Yes"])

    from datetime import datetime, timedelta
    start = datetime(2016, 1, 1)
    hours = days * 24
    temperatures = {}
    with (root / "weather.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["timestamp", "site_id", "airTemperature", "dewTemperature", "seaLvlPressure", "windSpeed"])
        for h in range(hours):
            timestamp = start + timedelta(hours=h)
            for site_n, site in enumerate(sites):
                temp = 16 + 9 * math.sin(2 * math.pi * (h % 24 - 7) / 24) + site_n + rng.gauss(0, .5)
                temperatures[(h, site_n)] = temp
                writer.writerow([timestamp, site, round(temp, 3), round(temp - 5, 3), 1012 + rng.gauss(0, 2), abs(rng.gauss(2, 1))])

    values = {b: [] for b in ids}
    for n, building in enumerate(ids):
        previous = 40 + n * 4
        area_scale = 1 + n * .12
        for h in range(hours):
            hour = h % 24; dow = (h // 24) % 7
            occupied = 1 if dow < 5 and 7 <= hour <= 19 else 0
            temp = temperatures[(h, n % 3)]
            demand = area_scale * (25 + 38 * occupied + 1.8 * max(temp - 19, 0))
            demand += 0.35 * previous + rng.gauss(0, 2)
            previous = max(demand, .1); values[building].append(round(previous, 3))
    with (root / "electricity_cleaned.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["timestamp", *ids])
        for h in range(hours):
            writer.writerow([start + timedelta(hours=h), *[values[b][h] for b in ids]])


if __name__ == "__main__":
    import sys
    generate(Path(sys.argv[1]))
