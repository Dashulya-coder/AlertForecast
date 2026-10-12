"""Origin frame: one row per forecast origin, with point-in-time features and (separately named) labels.

Two interval tables are used on purpose:
  - `state_iv` (Intervals.for_region_state): what a live system knew at t. Feeds features and the clear-at-t state.
  - `label_iv` (Intervals.for_region): the dataset's ends. Feeds labels only.
Labels are columns named y{h} and are kept apart from features by `FEATURE_COLUMNS`.
"""
import numpy as np
import pandas as pd

from alertforecast import config
from alertforecast.data import Intervals
from alertforecast.labels import make_label

FEATURE_COLUMNS = ["frac_alert_24h", "n_starts_24h", "hour", "dow"]
STATE_COLUMNS = ["clear"]


def build_frame(t, state_iv: Intervals, label_iv: Intervals = None):
    """Features at each origin t from `state_iv`. Labels from `label_iv` are added only when it is given."""
    t = np.asarray(t, dtype="int64")
    local = pd.to_datetime(t, unit="s", utc=True).tz_convert(config.KYIV_TZ)
    df = pd.DataFrame({"t": t})
    df["clear"] = ~state_iv.active_at(t)
    df["frac_alert_24h"] = state_iv.coverage_until(t, config.DAY_S) / config.DAY_S
    df["n_starts_24h"] = state_iv.starts_in_window(t, config.DAY_S)
    df["hour"] = local.hour.values  # Kyiv local time: DST shifts the calendar features
    df["dow"] = local.dayofweek.values
    df["day"] = local.strftime("%Y-%m-%d").values  # block-bootstrap id
    if label_iv is not None:
        for h in config.HORIZONS_H:
            df[f"y{h}"] = make_label(label_iv, t, h)
    return df
