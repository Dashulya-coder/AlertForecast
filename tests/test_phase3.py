import numpy as np
import pytest

from alertforecast import calibration, config, metrics, model, protocol, splits
from alertforecast.data import Intervals
from alertforecast.labels import make_label

H = 3600


def test_windows_are_contiguous_28_day_blocks_and_exclude_test():
    w = config.EVAL_WINDOWS
    assert w["w2"][1] == w["w1"][0] and w["w1"][1] == w["val"][0] and w["val"][1] == config.TEST_START
    for start, end in w.values():
        assert end - start == 28 * 86400  # no DST change in these months
        assert end <= config.TEST_START
    assert [len(splits.window_origins(n)) for n in ("val", "w1", "w2")] == [1344] * 3


def test_calibration_blocks_are_contiguous_nearest_first_and_end_at_the_window():
    for name, (start, _) in config.EVAL_WINDOWS.items():
        blocks = splits.calibration_blocks(name)
        assert len(blocks) == config.CAL_BLOCKS and blocks[0][1] == start
        for newer, older in zip(blocks, blocks[1:]):
            assert older[1] == newer[0]
        assert all(b - a >= 27 * 86400 for a, b in blocks)  # a block spanning a DST change is 1h off 28 days


@pytest.mark.parametrize("h", config.HORIZONS_H)
def test_calibration_origins_never_see_the_window(h):
    """Labels of calibration rows must not look into the window the calibrator is applied to."""
    rng = np.random.default_rng(3)
    window_start = config.EVAL_WINDOWS["val"][0]
    base = window_start - 130 * 86400
    starts = base + np.sort(rng.integers(0, 190 * 86400, 900))
    iv = Intervals.from_arrays(starts, starts + rng.integers(60, 180 * 60, 900))
    for block in splits.calibration_blocks("val"):
        t = protocol.calibration_origins(block, window_start, h)
        assert (t + h * H < window_start).all()
        full = make_label(iv, t, h)
        erased = make_label(iv.truncated_at(window_start - 1), t, h)
        assert (full == erased).all()


def test_recency_weights():
    t = np.array([100 * 86400, 70 * 86400, 40 * 86400])
    ref = 100 * 86400
    assert model.recency_weights(t, ref, None).tolist() == [1.0, 1.0, 1.0]
    w = model.recency_weights(t, ref, 30)
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(0.5) and w[2] == pytest.approx(0.25)
    assert (np.diff(w) < 0).all()  # older rows weigh less


def test_gbm_does_not_use_random_early_stopping():
    assert model.GBM_PARAMS["early_stopping"] is False


def test_stage_a_grid_is_preregistered():
    recipes = model.stage_a_recipes()
    assert len(recipes) == 8
    assert recipes[0].half_life_days is None and "n_starts_28d" not in recipes[0].features  # base first
    assert sorted({r.half_life_days or 0 for r in recipes}) == [0, 30, 60, 180]


def test_intercept_shift_matches_the_mean_outcome():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.5, 4000)
    y = rng.random(4000) < np.clip(p * 1.8, 0, 0.95)  # true rate is higher than predicted
    c = calibration.InterceptShift().fit(p, y)
    assert c.predict(p).mean() == pytest.approx(y.mean(), abs=1e-4)
    w = np.where(np.arange(4000) < 2000, 1.0, 0.0)
    cw = calibration.InterceptShift().fit(p, y, weights=w)
    assert cw.predict(p)[:2000].mean() == pytest.approx(y[:2000].mean(), abs=1e-3)


def test_platt_fixes_overconfidence_and_calibrators_are_monotone():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.05, 0.95, 6000)
    y = rng.random(6000) < true_p
    overconfident = calibration.sigmoid(2.5 * calibration.logit(true_p))
    c = calibration.Platt().fit(overconfident, y)
    assert metrics.brier(y, c.predict(overconfident)) < metrics.brier(y, overconfident)
    order = np.argsort(overconfident)
    for cal in calibration.stage_b_calibrators():
        out = cal.fit(overconfident, y).predict(overconfident)[order]
        assert (np.diff(out) >= -1e-12).all()  # monotone, so PR-AUC cannot change
    assert metrics.pr_auc(y, overconfident) == pytest.approx(metrics.pr_auc(y, c.predict(overconfident)))


def test_rows_requires_grid_alignment():
    import pandas as pd
    t = np.arange(0, 10 * config.GRID_S, config.GRID_S, dtype="int64")
    full = pd.DataFrame({"t": t, "v": np.arange(10)})
    assert protocol.rows(full, t[3:6])["v"].tolist() == [3, 4, 5]
    with pytest.raises(AssertionError):
        protocol.rows(full, np.array([t[1] + 7]))


def _two_region_table(naive_other=False):
    import pandas as pd
    T = config.VAL_START
    rows = [
        {"region": "Kyiv City", "started_at_s": T, "finished_at_s": T + 1800, "naive": False},
        {"region": "Cherkaska oblast", "started_at_s": T - 3600, "finished_at_s": T + 7200, "naive": False},
        {"region": "Cherkaska oblast", "started_at_s": T + 3 * H, "finished_at_s": T + 3 * H + 1800, "naive": naive_other},
    ]
    return T, pd.DataFrame(rows)


def test_neighbor_features_ignore_the_future():
    """Origin inside a running neighbour alert: moving its end or deleting later alerts changes nothing at t."""
    from alertforecast.neighbors import neighbor_features
    T, al = _two_region_table()
    t = np.array([T + 3600])  # Cherkaska is open since T-1h, its end (T+2h) is in the future
    base = neighbor_features(t, al, regions=["Cherkaska oblast"])
    assert base["nb_open_Cherkaska oblast"].tolist() == [1] and base["nb_age_Cherkaska oblast"].tolist() == [2.0]
    stretched = al.copy()
    stretched.loc[1, "finished_at_s"] = T + 9 * H  # different unknown end
    removed = al.drop(index=2)  # the later alert never happens
    for altered in (stretched, removed):
        assert neighbor_features(t, altered, regions=["Cherkaska oblast"]).equals(base)


def test_neighbor_naive_alert_is_open_in_the_state_view():
    from alertforecast.neighbors import neighbor_features
    T, al = _two_region_table(naive_other=False)
    al.loc[1, ["finished_at_s", "naive"]] = [T - 3600 + 1800, True]  # first Cherkaska alert is naive
    t = np.array([T + 3600])  # after its imputed end, before the next Cherkaska activation at T+3h
    f = neighbor_features(t, al, regions=["Cherkaska oblast"])
    assert f["nb_open_Cherkaska oblast"].tolist() == [1]  # still open: no end message, no 1800s signature
    assert f["nb_n_open"].tolist() == [1]


def test_multi_window_diff_ci():
    rng = np.random.default_rng(4)
    wins_a, wins_b = [], []
    for _ in range(3):
        days = np.repeat(np.arange(28), 48)
        y = rng.random(len(days)) < 0.3
        good = np.clip(y * 0.5 + rng.random(len(days)) * 0.4, 0, 1)
        wins_a.append((y, good, days))
        wins_b.append((y, rng.random(len(days)), days))
    same = metrics.multi_window_diff_ci(metrics.brier, wins_a, wins_a, n_boot=50)
    assert same == (0.0, 0.0, 0.0)
    d, lo, hi = metrics.multi_window_diff_ci(metrics.brier, wins_a, wins_b, n_boot=100)
    assert d < 0 and hi < 0  # informative predictions have a lower Brier than noise
