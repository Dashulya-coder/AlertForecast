"""Origin frame: one row per forecast origin, with point-in-time features and (separately named) labels.

Two interval tables are used on purpose:
  - `state_iv` (Intervals.for_region_state): what a live system knew at t. Feeds features and the clear-at-t state.
  - `label_iv` (Intervals.for_region): the dataset's ends. Feeds labels only.
Labels are columns named y{h} and are kept apart from features by the feature lists below.
"""
import numpy as np
import pandas as pd

from alertforecast import config
from alertforecast.data import Intervals
from alertforecast.labels import make_label

HOUR, DAY = config.HOUR_S, config.DAY_S

# Feature groups used by the models (Stage A: base vs base + recent-rate).
BASE_FEATURES = [
    "hour", "dow", "open", "open_age_h", "since_last_start_h", "since_last_end_h",
    "n_starts_1h", "n_starts_3h", "n_starts_6h", "n_starts_24h", "frac_alert_6h", "frac_alert_24h",
]
RATE_FEATURES = ["n_starts_7d", "n_starts_28d", "frac_alert_7d", "frac_alert_28d"]
FEATURE_COLUMNS = BASE_FEATURES + RATE_FEATURES
STATE_COLUMNS = ["clear"]


def build_frame(t, state_iv: Intervals, label_iv: Intervals = None):
    """Features at each origin t from `state_iv`. Labels from `label_iv` are added only when it is given."""
    t = np.asarray(t, dtype="int64")
    local = pd.to_datetime(t, unit="s", utc=True).tz_convert(config.KYIV_TZ)
    df = pd.DataFrame({"t": t})
    active = state_iv.active_at(t)
    df["clear"] = ~active
    df["open"] = active.astype(int)
    df["open_age_h"] = state_iv.open_age(t) / HOUR
    df["since_last_start_h"] = state_iv.since_last_start(t) / HOUR
    df["since_last_end_h"] = state_iv.since_last_end(t) / HOUR
    for name, w in (("1h", HOUR), ("3h", 3 * HOUR), ("6h", 6 * HOUR), ("24h", DAY), ("7d", 7 * DAY), ("28d", 28 * DAY)):
        df[f"n_starts_{name}"] = state_iv.starts_in_window(t, w)
    for name, w in (("6h", 6 * HOUR), ("24h", DAY), ("7d", 7 * DAY), ("28d", 28 * DAY)):
        df[f"frac_alert_{name}"] = state_iv.coverage_until(t, w) / w
    df["hour"] = local.hour.values  # Kyiv local time: DST shifts the calendar features
    df["dow"] = local.dayofweek.values
    df["day"] = local.strftime("%Y-%m-%d").values  # block-bootstrap id
    if label_iv is not None:
        for h in config.HORIZONS_H:
            df[f"y{h}"] = make_label(label_iv, t, h)
    return df
