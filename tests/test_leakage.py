"""Leakage tests: nothing after the forecast time may change what the model sees or is fit on."""
import numpy as np
import pandas as pd
import pytest

from alertforecast import config, splits
from alertforecast.data import Intervals, complete_until
from alertforecast.frame import FEATURE_COLUMNS, STATE_COLUMNS, build_frame
from alertforecast.labels import make_label
from conftest import H, cutoff_inside_alert, random_intervals


def perturb_after(iv: Intervals, cutoff, seed):
    """Keep everything up to the cutoff, but rewrite the future: stretch the alert running at the cutoff,
    delete the real later alerts and add random new ones."""
    rng = np.random.default_rng(seed)
    starts, ends = iv.starts.copy(), iv.ends.copy()
    running = (starts <= cutoff) & (ends > cutoff)
    ends[running] += 3 * H  # same start, different (unknown at cutoff) end
    keep = starts <= cutoff
    new_s = cutoff + 1 + np.sort(rng.integers(0, 20 * 86400, 80))
    new_e = new_s + rng.integers(60, 120 * 60, 80)
    return Intervals.from_arrays(np.concatenate([starts[keep], new_s]), np.concatenate([ends[keep], new_e]))


@pytest.mark.parametrize("seed,n", [(0, 400), (1, 400), (2, 60), (3, 60)])  # dense and sparse alert patterns
def test_features_at_t_ignore_the_future(seed, n):
    iv, base, span = random_intervals(seed, n=n)
    cutoff = cutoff_inside_alert(iv, after=base + 60 * 86400)  # an alert is running at the cutoff, with an unknown end
    assert iv.active_at([cutoff])[0]
    t = np.arange(base + 2 * 86400, cutoff + 1, 1800, dtype="int64")  # every origin at or before the cutoff
    original = build_frame(t, iv)
    for altered_iv in (iv.truncated_at(cutoff), perturb_after(iv, cutoff, seed)):
        altered = build_frame(t, altered_iv)
        cols = FEATURE_COLUMNS + STATE_COLUMNS
        assert original[cols].equals(altered[cols])


def test_perturbation_has_teeth():
    """The same perturbation does change labels for origins near the cutoff, so the test above is not vacuous."""
    iv, base, _ = random_intervals(0, n=20)  # sparse, so labels near the cutoff are not all already 1
    cutoff = base + 60 * 86400
    altered = perturb_after(iv, cutoff, 0)
    t = np.arange(cutoff - 6 * H, cutoff + 1, 1800, dtype="int64")
    assert (make_label(iv, t, 6) != make_label(altered, t, 6)).any()


# ---- the naive flag: end = start + 30 min is imputed from the future (next activation > 3h later) ----

def naive_table(extra_rows=()):
    """Region 'Kyiv City': naive alert at T (imputed end T+30min), next alert at T+5h."""
    T = config.VAL_START
    rows = [
        {"region": "Kyiv City", "started_at_s": T, "finished_at_s": T + 1800, "naive": True},
        {"region": "Kyiv City", "started_at_s": T + 5 * H, "finished_at_s": T + 5 * H + 900, "naive": False},
        *extra_rows,
    ]
    return T, pd.DataFrame(rows)


def test_naive_alert_stays_open_in_the_real_time_state():
    T, al = naive_table()
    state, dataset = Intervals.for_region_state(al), Intervals.for_region(al)
    t = np.array([T + 3600, T + 2 * H, T + 5 * H - 1])  # after the imputed end, before the next activation
    assert state.active_at(t).tolist() == [True, True, True]  # no end message yet: open
    assert dataset.active_at(t).tolist() == [False, False, False]  # the dataset's imputed end would say "clear"
    assert state.active_at([T + 5 * H + 1800]).tolist() == [False]  # after the next alert closed


def test_naive_alert_features_cannot_reveal_the_next_gap():
    """At an origin inside the naive shadow, a naive alert and a genuine alert whose end message has not arrived
    yet look identical. Using the dataset's imputed end would tell them apart, which leaks 'no alert for 3h'."""
    T, naive_al = naive_table()
    t = np.array([T + 2 * H])
    genuine_al = naive_al.copy()
    genuine_al.loc[0, ["finished_at_s", "naive"]] = [T + 2 * H + 600, False]  # real end comes later than t
    f_naive = build_frame(t, Intervals.for_region_state(naive_al))
    f_genuine = build_frame(t, Intervals.for_region_state(genuine_al))
    cols = FEATURE_COLUMNS + STATE_COLUMNS
    assert f_naive[cols].equals(f_genuine[cols])
    # the same comparison with the dataset's ends differs: this is the leak that the state view removes
    g_naive = build_frame(t, Intervals.for_region(naive_al))
    g_genuine = build_frame(t, Intervals.for_region(genuine_al))
    assert not g_naive[cols].equals(g_genuine[cols])


def test_state_features_ignore_when_the_next_activation_comes():
    """Moving the next activation later (still after t) does not change state features at t."""
    T, al = naive_table()
    later = al.copy()
    later.loc[1, ["started_at_s", "finished_at_s"]] = [T + 9 * H, T + 9 * H + 900]
    t = np.array([T + 2 * H, T + 3 * H])
    cols = FEATURE_COLUMNS + STATE_COLUMNS
    assert build_frame(t, Intervals.for_region_state(al))[cols].equals(build_frame(t, Intervals.for_region_state(later))[cols])


def test_complete_until_marks_where_the_snapshot_is_trustworthy():
    T = 1_000_000
    naive_last = pd.DataFrame([{"region": "Kyiv City", "started_at_s": T, "finished_at_s": T + 1800, "naive": True}])
    assert complete_until(naive_last) == T + 3 * H  # successor is open and cannot have started earlier
    closed_last = pd.DataFrame([{"region": "Kyiv City", "started_at_s": T, "finished_at_s": T + 600, "naive": False}])
    assert complete_until(closed_last) == T + 600  # anything later may be an open alert missing from the file


def test_eval_origins_are_cut_where_labels_are_unreliable():
    cu = config.TEST_END - 5 * H
    full = splits.eval_origins("test", allow_test=True)
    for h in config.HORIZONS_H:
        kept = splits.eval_origins("test", allow_test=True, horizon_h=h, complete_until=cu)
        dropped = np.setdiff1d(full, kept)
        assert len(dropped) > 0
        assert (kept + h * H <= cu).all()  # every kept label window ends inside the recorded data
        assert (dropped + h * H > cu).all()  # and exactly the others are dropped


# ---- purge boundary (strict) ----

@pytest.mark.parametrize("h", config.HORIZONS_H)
def test_fit_rows_are_purged_strictly_per_horizon(h):
    eval_start = config.VAL_START
    t = splits.fit_origins("val", h)
    assert t[0] == config.TRAIN_ORIGIN_START
    assert t[-1] + h * H < eval_start  # strictly before the first evaluation origin
    assert t[-1] + h * H + config.GRID_S == eval_start  # and only one grid step of slack
    assert (np.diff(t) == config.GRID_S).all()


def test_fit_for_test_uses_train_plus_val_and_purges_before_test():
    t = splits.fit_origins("test", 6)
    assert t[-1] + 6 * H < config.TEST_START
    assert t[-1] > config.VAL_START  # val origins are used for the final fit


@pytest.mark.parametrize("h", config.HORIZONS_H)
def test_fit_labels_do_not_depend_on_the_evaluation_period(h):
    """Labels of all fit rows are identical if every alert at or after the evaluation start is erased."""
    base = config.VAL_START - 90 * 86400
    rng = np.random.default_rng(7)
    starts = base + np.sort(rng.integers(0, 120 * 86400, 600))
    iv = Intervals.from_arrays(starts, starts + rng.integers(60, 180 * 60, 600))
    t = splits.fit_origins("val", h)
    t = t[t >= base]
    full = make_label(iv, t, h)
    erased = make_label(iv.truncated_at(config.VAL_START - 1), t, h)  # nothing from the evaluation period survives
    assert (full == erased).all()


@pytest.mark.parametrize("h", config.HORIZONS_H)
def test_boundary_row_would_leak_and_is_excluded(h):
    """An alert starting exactly at the first evaluation origin is in the label window of the row at t = eval_start - h."""
    eval_start = config.VAL_START
    iv = Intervals.from_arrays([eval_start], [eval_start + 1800])
    t_boundary = np.array([eval_start - h * H])
    assert make_label(iv, t_boundary, h)[0]  # the label looks exactly at the evaluation start
    assert t_boundary[0] not in splits.fit_origins("val", h)  # the strict purge excludes it
    t_late = np.array([eval_start - h * H + config.GRID_S])
    assert t_late[0] not in splits.fit_origins("val", h)


def test_test_split_is_locked():
    with pytest.raises(PermissionError):
        splits.eval_origins("test")
    assert len(splits.eval_origins("test", allow_test=True)) == 28 * 48


def test_val_origins_are_28_full_days():
    t = splits.eval_origins("val")
    assert len(t) == 28 * 48
    assert t[0] == config.VAL_START and t[-1] == config.TEST_START - config.GRID_S
