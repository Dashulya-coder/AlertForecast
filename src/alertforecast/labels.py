"""Labels. These use the alert table over (t, t + h] and must never be used as features."""
import numpy as np

from alertforecast import config
from alertforecast.data import Intervals


def make_label(iv: Intervals, t, h_hours):
    """1 if any alert overlaps the window (t, t + h], including one already active at t.

    An alert [s, e) overlaps when s <= t + h and e > t. An alert starting exactly at t + h counts.
    """
    t = np.asarray(t, dtype="int64")
    idx = iv._first_ending_after(t)  # first alert whose end is after t
    ok = idx < len(iv.starts)
    out = np.zeros(len(t), dtype=bool)
    out[ok] = iv.starts[idx[ok]] <= t[ok] + h_hours * config.HOUR_S
    return out
