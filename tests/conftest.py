import numpy as np
import pytest

from alertforecast.data import Intervals

H = 3600


def random_intervals(seed, n=400, span_days=120, base=1_700_000_000):
    """Synthetic alerts: random starts, durations from 1 to 180 minutes. Overlaps are merged by Intervals."""
    rng = np.random.default_rng(seed)
    starts = base + np.sort(rng.integers(0, span_days * 86400, n))
    ends = starts + rng.integers(60, 180 * 60, n)
    return Intervals.from_arrays(starts, ends), base, span_days


def cutoff_inside_alert(iv, after, min_len_s=3600):
    """A cutoff in the middle of an alert that is long enough to contain several 30-minute origins.

    Leaks through alert end times only show up when an alert is running at the cutoff.
    """
    long_enough = np.flatnonzero((iv.starts > after) & (iv.ends - iv.starts >= min_len_s))
    k = long_enough[0]
    return int((iv.starts[k] + iv.ends[k]) // 2)


@pytest.fixture
def toy():
    """Alert A: T+2h to T+2.5h. Alert B: T+10h to T+11h."""
    T = 1_000_000 * 3600
    return T, Intervals.from_arrays([T + 2 * H, T + 10 * H], [T + int(2.5 * H), T + 11 * H])
