"""Alert table loading and point-in-time alert state.

Alerts are half-open intervals [start, end) in UTC epoch seconds. Overlapping intervals of one region are merged,
so starts and ends are both sorted.

Leakage rule 1 (CLAUDE.md): everything the model sees at origin t must be computable from alerts with start <= t,
with ends treated as "unknown" if end > t. `Intervals.active_at` and `Intervals.coverage_until` respect this.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alertforecast import config


def load_alerts(path=None):
    """All volunteer alerts, all regions, with epoch-second start/end columns (UTC)."""
    df = pd.read_csv(path or config.VOLUNTEER_CSV, parse_dates=["started_at", "finished_at"])
    for col in ("started_at", "finished_at"):
        df[col + "_s"] = df[col].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[s]").astype("int64")
    return df


NAIVE_RULE_S = 3 * 3600  # the dataset only writes naive=True when the next activation is > 3h after the start
OPEN_END = 4_000_000_000  # "still open, end unknown" (year 2096)


def complete_until(alerts, region=config.KYIV_REGION):
    """Latest instant up to which the region's alerts are fully recorded in this snapshot.

    The dataset writes an alert only when it closes. If the region's last row is naive, its successor is open
    and absent, and cannot have started before last_start + 3h (that is what makes the row naive). Otherwise
    anything after the last row's end may be missing. Labels are only trustworthy for windows ending before this.
    """
    last = alerts[alerts.region == region].sort_values("started_at_s").iloc[-1]
    if last.naive:
        return int(last.started_at_s + NAIVE_RULE_S)
    return int(last.finished_at_s)


@dataclass(frozen=True)
class Intervals:
    """Sorted, non-overlapping alert intervals [start, end). Only strictly overlapping intervals are merged."""

    starts: np.ndarray
    ends: np.ndarray

    @classmethod
    def from_arrays(cls, starts, ends):
        starts, ends = np.asarray(starts, dtype="int64"), np.asarray(ends, dtype="int64")
        if len(starts) == 0:
            return cls(starts, ends)
        order = np.argsort(starts, kind="stable")
        s, e = starts[order], ends[order]
        out_s, out_e = [s[0]], [e[0]]
        for a, b in zip(s[1:], e[1:]):
            if a < out_e[-1]:
                out_e[-1] = max(out_e[-1], b)
            else:
                out_s.append(a)
                out_e.append(b)
        return cls(np.array(out_s, dtype="int64"), np.array(out_e, dtype="int64"))

    @classmethod
    def for_region(cls, alerts, region=config.KYIV_REGION):
        """Dataset intervals with the dataset's ends. Use for LABELS only: a naive alert's end (start + 30 min) is
        imputed from the future (the next activation > 3h later), so it must not feed features or the clear state."""
        r = alerts[alerts.region == region]
        return cls.from_arrays(r.started_at_s.values, r.finished_at_s.values)

    @classmethod
    def for_region_state(cls, alerts, region=config.KYIV_REGION):
        """Intervals as a live system would have seen them. Use for FEATURES and the clear-at-t state.

        A naive alert never got an end message, so in real time it stays open (end unknown) until the next
        activation arrives; its end is therefore the next start in the region (or open if there is none yet).
        """
        r = alerts[alerts.region == region].sort_values("started_at_s")
        starts = r.started_at_s.values
        ends = r.finished_at_s.values.copy()
        nxt = np.append(starts[1:], OPEN_END)
        naive = r.naive.values.astype(bool)
        ends[naive] = nxt[naive]
        return cls.from_arrays(starts, ends)

    def __len__(self):
        return len(self.starts)

    def _first_ending_after(self, t):
        return np.searchsorted(self.ends, t, side="right")

    def active_at(self, t):
        """True where an alert has started (start <= t) and has not ended (end > t)."""
        t = np.asarray(t, dtype="int64")
        idx = self._first_ending_after(t)
        ok = idx < len(self.starts)
        out = np.zeros(len(t), dtype=bool)
        out[ok] = self.starts[idx[ok]] <= t[ok]
        return out

    def _cumulative_coverage(self, x):
        """Seconds under alert in (-inf, x], with every alert's end clipped at x (so it only uses start <= x)."""
        x = np.asarray(x, dtype="int64")
        lengths = self.ends - self.starts
        cum = np.concatenate([[0], np.cumsum(lengths)])
        j = np.searchsorted(self.starts, x, side="right")  # alerts with start <= x
        out = np.zeros(len(x), dtype="int64")
        has = j > 0
        last = j[has] - 1
        out[has] = cum[last] + np.minimum(x[has], self.ends[last]) - self.starts[last]
        return out

    def coverage_until(self, t, window_s):
        """Seconds under alert in (t - window, t], using only information available at t."""
        t = np.asarray(t, dtype="int64")
        return self._cumulative_coverage(t) - self._cumulative_coverage(t - window_s)

    def starts_in_window(self, t, window_s):
        """Number of alert starts in (t - window, t]."""
        t = np.asarray(t, dtype="int64")
        return np.searchsorted(self.starts, t, side="right") - np.searchsorted(self.starts, t - window_s, side="right")

    def truncated_at(self, cutoff):
        """What the table looks like if everything after `cutoff` is unknown. Used by leakage tests.

        Alerts starting after the cutoff vanish. An alert still running at the cutoff is kept as running
        (end = cutoff + 1, so it is active at the cutoff); its true end is unknown.
        """
        keep = self.starts <= cutoff
        return Intervals(self.starts[keep], np.minimum(self.ends[keep], cutoff + 1))
