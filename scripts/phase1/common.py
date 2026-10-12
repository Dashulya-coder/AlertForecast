"""Shared helpers for phase 1: data loading, fixed split dates, label construction."""
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOLUNTEER_CSV = os.path.join(ROOT, "data", "raw", "vadimkin_2026-10-12", "volunteer_data_en.csv")

KYIV_TZ = ZoneInfo("Europe/Kyiv")
HORIZONS_H = (1, 3, 6)
GRID_S = 30 * 60


def kyiv_midnight_utc(year, month, day):
    """Start of a Europe/Kyiv local day as a UTC epoch second."""
    return int(pd.Timestamp(year=year, month=month, day=day, tz=KYIV_TZ).tz_convert("UTC").timestamp())


# Fixed split dates, see CLAUDE.md. Test end is exclusive.
VAL_START = kyiv_midnight_utc(2026, 8, 17)
TEST_START = kyiv_midnight_utc(2026, 9, 14)
TEST_END = kyiv_midnight_utc(2026, 10, 12)
# Provisional: first train origin. The first days of the war are sparse and unusual.
TRAIN_ORIGIN_START = int(pd.Timestamp("2022-03-01", tz="UTC").timestamp())


def load_alerts():
    """All volunteer alerts with epoch-second start/end columns (UTC)."""
    df = pd.read_csv(VOLUNTEER_CSV, parse_dates=["started_at", "finished_at"])
    for col in ("started_at", "finished_at"):
        df[col + "_s"] = df[col].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[s]").astype("int64")
    return df


def merged_intervals(starts, ends):
    """Merge overlapping or touching intervals. Returns sorted (starts, ends) arrays with ends sorted."""
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    out_s, out_e = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= out_e[-1]:
            out_e[-1] = max(out_e[-1], b)
        else:
            out_s.append(a)
            out_e.append(b)
    return np.array(out_s), np.array(out_e)


def first_ending_after(starts, ends, t):
    """Index of the first merged interval whose end is after t (ends must be sorted)."""
    return np.searchsorted(ends, t, side="right")


def active_at(starts, ends, t):
    """True where an interval is active at t, meaning started at or before t and ends after t."""
    idx = first_ending_after(starts, ends, t)
    ok = idx < len(starts)
    out = np.zeros(len(t), dtype=bool)
    out[ok] = starts[idx[ok]] <= t[ok]
    return out


def label(starts, ends, t, h_hours):
    """1 if any interval overlaps (t, t + h]. Uses only the alert table, so it is the label, never a feature."""
    idx = first_ending_after(starts, ends, t)
    ok = idx < len(starts)
    out = np.zeros(len(t), dtype=bool)
    out[ok] = starts[idx[ok]] < t[ok] + h_hours * 3600
    return out


def origins(start_s, end_s):
    """30-minute UTC grid in [start_s, end_s)."""
    return np.arange(start_s, end_s, GRID_S, dtype="int64")
