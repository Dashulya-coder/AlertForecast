import numpy as np
import pandas as pd
import pytest

from alertforecast import metrics
from alertforecast.baselines import Climatology, Persistence, RecentActivity


def test_brier_and_skill():
    y, p = np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.5, 0.5])
    assert metrics.brier(y, p) == pytest.approx((0.01 + 0.01 + 0.25 + 0.25) / 4)
    assert metrics.brier_skill_score(y, p, np.full(4, 0.5)) == pytest.approx(1 - 0.13 / 0.25)
    assert metrics.brier_skill_score(y, y.astype(float), y.astype(float)) != metrics.brier_skill_score(y, p, p)


def test_pr_auc_constant_predictor_equals_base_rate():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.2
    assert metrics.pr_auc(y, np.zeros(5000)) == pytest.approx(y.mean())
    assert metrics.pr_auc(y, y.astype(float)) == pytest.approx(1.0)
    assert np.isnan(metrics.pr_auc(np.zeros(10, dtype=bool), np.ones(10)))


def test_precision_recall_and_threshold():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    assert metrics.precision_recall_at(y, p, 0.65) == (1.0, 0.75)
    thr = metrics.threshold_for_recall(y, p, 0.75)
    assert thr == 0.7
    assert metrics.precision_recall_at(y, p, thr)[1] >= 0.75
    assert metrics.threshold_for_recall(y, p, 1.0) == 0.2


def test_thresholds_for_recall_targets_are_monotone_and_reach_the_target():
    rng = np.random.default_rng(5)
    y = rng.random(3000) < 0.25
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.25, 3000), 0, 1)
    thr = metrics.thresholds_for_recalls(y, p)
    assert list(thr) == [0.5, 0.8, 0.9]
    assert thr[0.5] >= thr[0.8] >= thr[0.9]  # a higher recall target needs a lower threshold
    for target, t in thr.items():
        precision, recall = metrics.precision_recall_at(y, p, t)
        assert recall >= target and 0 < precision < 1
    assert all(np.isnan(v) for v in metrics.thresholds_for_recalls(y, np.full(len(y), 0.3)).values())


def test_log_loss_is_finite_for_hard_predictions():
    assert np.isfinite(metrics.log_loss(np.array([1, 0]), np.array([0.0, 1.0])))


def test_block_bootstrap_is_deterministic_and_ordered():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(28), 48)
    y = rng.random(len(days)) < 0.3
    p = np.clip(y * 0.4 + rng.random(len(days)) * 0.5, 0, 1)
    a = metrics.block_bootstrap_ci(metrics.pr_auc, y, p, days, n_boot=200, seed=3)
    b = metrics.block_bootstrap_ci(metrics.pr_auc, y, p, days, n_boot=200, seed=3)
    assert a == b and a[0] < metrics.pr_auc(y, p) < a[1]


def test_paired_bootstrap_diff():
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(28), 48)
    y = rng.random(len(days)) < 0.3
    good = np.clip(y * 0.5 + rng.random(len(days)) * 0.4, 0, 1)
    noise = rng.random(len(days))
    same = metrics.block_bootstrap_diff_ci(metrics.pr_auc, y, good, good, days, n_boot=100)
    assert same == (0.0, 0.0, 0.0)  # identical predictions: no difference at all
    d, lo, hi = metrics.block_bootstrap_diff_ci(metrics.pr_auc, y, good, noise, days, n_boot=200)
    assert d > 0 and lo > 0  # informative predictions clearly beat noise


def test_persistence_is_degenerate_in_view_b():
    frame = pd.DataFrame({"clear": [True, True, True]})
    assert Persistence().predict(frame).tolist() == [0.0, 0.0, 0.0]
    assert Persistence().predict(pd.DataFrame({"clear": [False, True]})).tolist() == [1.0, 0.0]


def test_climatology_uses_only_training_rows():
    train = pd.DataFrame({"hour": [0, 0, 1, 1]})
    y = np.array([1, 1, 0, 0])
    c = Climatology(("hour",), prior_strength=0.0).fit(train, y)
    assert c.predict(pd.DataFrame({"hour": [0, 1, 5]})).tolist() == [1.0, 0.0, 0.5]  # unseen cell -> global rate


def test_recent_activity_learns_a_monotone_signal():
    rng = np.random.default_rng(0)
    frac = rng.random(4000) * 0.3
    n = rng.integers(0, 8, 4000)
    y = rng.random(4000) < (0.05 + 2 * frac)
    frame = pd.DataFrame({"frac_alert_24h": frac, "n_starts_24h": n})
    m = RecentActivity().fit(frame, y)
    lo = m.predict(pd.DataFrame({"frac_alert_24h": [0.0], "n_starts_24h": [1]}))[0]
    hi = m.predict(pd.DataFrame({"frac_alert_24h": [0.3], "n_starts_24h": [1]}))[0]
    assert hi > lo
