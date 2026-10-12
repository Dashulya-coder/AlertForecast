"""Metrics: PR-AUC, Brier (and skill score), log loss, precision/recall at a frozen threshold, calibration,
and a block bootstrap by day.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

LOG_LOSS_EPS = 1e-4  # same clipping for every model so log loss stays finite for 0/1 predictions


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def brier_skill_score(y, p, p_ref):
    """1 - Brier(model) / Brier(reference). Positive means better than the reference."""
    ref = brier(y, p_ref)
    return float("nan") if ref == 0 else 1.0 - brier(y, p) / ref


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pr_auc(y, p):
    """Average precision. Equals the base rate for a constant predictor; undefined without positives."""
    y = np.asarray(y)
    return float("nan") if y.sum() == 0 else float(average_precision_score(y, p))


def precision_recall_at(y, p, threshold):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(p) >= threshold
    tp = int((pred & y).sum())
    precision = tp / pred.sum() if pred.sum() else float("nan")
    recall = tp / y.sum() if y.sum() else float("nan")
    return float(precision), float(recall)


def threshold_for_recall(y, p, target_recall):
    """Highest threshold whose recall is >= target. Choose this on validation only, then freeze it."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    pos = np.sort(p[y])[::-1]
    if len(pos) == 0:
        return float("nan")
    k = int(np.ceil(target_recall * len(pos)))  # need at least k positives at or above the threshold
    return float(pos[k - 1])


RECALL_TARGETS = (0.5, 0.8, 0.9)  # 0.8 is the main operating point


def thresholds_for_recalls(y, p, targets=RECALL_TARGETS):
    """{target recall: threshold}, chosen on the given data (validation). Constant predictors have no useful
    threshold, so they get NaN."""
    p = np.asarray(p, dtype=float)
    if len(np.unique(p)) == 1:
        return {r: float("nan") for r in targets}
    return {r: threshold_for_recall(y, p, r) for r in targets}


def calibration_table(y, p, n_bins=10):
    """Reliability table over equal-count bins."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    if df.p.nunique() <= 1:
        return df.agg({"p": "mean", "y": "mean"}).to_frame().T.assign(n=len(df))
    df["bin"] = pd.qcut(df.p.rank(method="first"), min(n_bins, len(df)), labels=False)
    return df.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).reset_index(drop=True)


def block_bootstrap_diff_ci(metric_fn, y, p_a, p_b, day_ids, n_boot=500, seed=0, alpha=0.05):
    """CI for metric(y, p_a) - metric(y, p_b) on the SAME resampled days (paired), so shared noise cancels."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    day_ids = np.asarray(day_ids)
    days = np.unique(day_ids)
    idx_by_day = {d: np.flatnonzero(day_ids == d) for d in days}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_day[d] for d in rng.choice(days, size=len(days), replace=True)])
        v = metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])
        if not np.isnan(v):
            vals.append(v)
    point = metric_fn(y, p_a) - metric_fn(y, p_b)
    if not vals:
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def block_bootstrap_ci(metric_fn, y, p, day_ids, n_boot=500, seed=0, alpha=0.05):
    """CI for metric_fn(y, p) resampling whole days with replacement, because alerts cluster in time."""
    y, p = np.asarray(y), np.asarray(p)
    day_ids = np.asarray(day_ids)
    days = np.unique(day_ids)
    idx_by_day = {d: np.flatnonzero(day_ids == d) for d in days}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in pick])
        v = metric_fn(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
