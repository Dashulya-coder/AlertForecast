import numpy as np

from alertforecast.data import Intervals
from alertforecast.labels import make_label
from conftest import H, random_intervals


def test_active_at_half_open(toy):
    T, iv = toy
    t = np.array([T + 2 * H - 1, T + 2 * H, T + int(2.5 * H) - 1, T + int(2.5 * H)])
    assert iv.active_at(t).tolist() == [False, True, True, False]


def test_label_conventions(toy):
    T, iv = toy
    # alert starts exactly at t + h: counts, the window is (t, t + h]
    assert make_label(iv, [T + 1 * H], 1).tolist() == [True]
    # nothing within 1h from T
    assert make_label(iv, [T], 1).tolist() == [False]
    # an alert already running at t counts
    assert make_label(iv, [T + int(2.25 * H)], 1).tolist() == [True]
    # alert ended exactly at t: does not count; next alert is 7.5h away
    assert make_label(iv, [T + int(2.5 * H)], 6).tolist() == [False]
    assert make_label(iv, [T + int(2.5 * H)], 8).tolist() == [True]  # 8h window reaches T+10.5h, alert B starts T+10h
    assert make_label(iv, [T + 3 * H], 6).tolist() == [False]
    assert make_label(iv, [T + 5 * H], 6).tolist() == [True]


def test_coverage_clips_at_origin(toy):
    T, iv = toy
    # at T+2.25h only 0.25h of alert A is known, although A really lasts until T+2.5h
    assert iv.coverage_until([T + int(2.25 * H)], H).tolist() == [int(0.25 * H)]
    assert iv.coverage_until([T + 3 * H], H).tolist() == [int(0.5 * H)]
    assert iv.coverage_until([T], H).tolist() == [0]


def test_starts_in_window(toy):
    T, iv = toy
    assert iv.starts_in_window([T + 2 * H], H).tolist() == [1]  # (T+1h, T+2h] includes the start at T+2h
    assert iv.starts_in_window([T + 11 * H], 24 * H).tolist() == [2]
    assert iv.starts_in_window([T + 1 * H], 24 * H).tolist() == [0]


def test_label_matches_brute_force():
    iv, base, span = random_intervals(1)
    rng = np.random.default_rng(2)
    t = base + rng.integers(0, span * 86400, 2000)
    for h in (1, 3, 6):
        fast = make_label(iv, t, h)
        slow = np.array([np.any((iv.starts <= ti + h * H) & (iv.ends > ti)) for ti in t])
        assert (fast == slow).all()


def test_coverage_and_active_match_brute_force():
    iv, base, span = random_intervals(3)
    rng = np.random.default_rng(4)
    t = base + rng.integers(0, span * 86400, 1500)
    window = 24 * H
    fast = iv.coverage_until(t, window)
    slow = np.array([np.clip(np.minimum(iv.ends, ti) - np.maximum(iv.starts, ti - window), 0, None).sum() for ti in t])
    assert (fast == slow).all()
    active_slow = np.array([np.any((iv.starts <= ti) & (iv.ends > ti)) for ti in t])
    assert (iv.active_at(t) == active_slow).all()


def test_merge_overlapping():
    iv = Intervals.from_arrays([0, 5, 20], [10, 8, 30])
    assert iv.starts.tolist() == [0, 20] and iv.ends.tolist() == [10, 30]
