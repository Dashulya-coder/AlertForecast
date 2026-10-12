"""Stage C feature group: real-time state of the other regions.

Per region: nb_open_<r> (0/1) and nb_age_<r> (hours since the start of the open alert, 0 if clear), plus nb_n_open.
Built from Intervals.for_region_state, so a region's naive alert stays open until that region's next activation and
nothing after t is used (same guarantees as the Kyiv features, tested in tests/test_phase3.py).
"""
import numpy as np
import pandas as pd

from alertforecast import config
from alertforecast.data import Intervals

MIN_ALERTS = 100  # regions with fewer alerts in the file are ignored (Luhanska oblast has 3)


def neighbor_regions(alerts):
    counts = alerts.region.value_counts()
    return sorted(r for r in counts.index if counts[r] >= MIN_ALERTS and r != config.KYIV_REGION)


def neighbor_features(t, alerts, regions=None):
    t = np.asarray(t, dtype="int64")
    regions = regions if regions is not None else neighbor_regions(alerts)
    cols = {}
    n_open = np.zeros(len(t), dtype=int)
    for r in regions:
        iv = Intervals.for_region_state(alerts, r)
        active = iv.active_at(t)
        cols[f"nb_open_{r}"] = active.astype(int)
        cols[f"nb_age_{r}"] = iv.open_age(t) / config.HOUR_S
        n_open += active
    cols["nb_n_open"] = n_open
    return pd.DataFrame(cols)


def neighbor_columns(regions):
    return [f"nb_open_{r}" for r in regions] + [f"nb_age_{r}" for r in regions] + ["nb_n_open"]
