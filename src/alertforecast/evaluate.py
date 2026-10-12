"""Evaluate predictions in both views.

View (a): all origins. View (b): origins where Kyiv is clear at t, as a live system would know it
(naive alerts count as open, see data.Intervals.for_region_state).
Each view is scored against a reference (climatology) fitted for that same view.
"""
import numpy as np
import pandas as pd

from alertforecast import metrics

MAIN_RECALL = 0.8


def evaluate_view(frame, y, p, p_ref, thresholds, n_boot=500, seed=0):
    """Metric dictionary for one view. `frame` supplies the day ids for the block bootstrap.

    `thresholds` is {target recall: threshold}, chosen on validation. On validation itself this is in-sample;
    for test the same thresholds must be passed in frozen. Columns precision/recall are the MAIN_RECALL point;
    precision_r50 / precision_r90 etc. are the other operating points, each with the recall actually achieved.
    """
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    days = frame["day"].values
    out = {"n": len(y), "base_rate": float(y.mean())}
    out["pr_auc"] = metrics.pr_auc(y, p)
    out["pr_auc_lo"], out["pr_auc_hi"] = metrics.block_bootstrap_ci(metrics.pr_auc, y, p, days, n_boot=n_boot, seed=seed)
    out["brier"] = metrics.brier(y, p)
    out["brier_lo"], out["brier_hi"] = metrics.block_bootstrap_ci(metrics.brier, y, p, days, n_boot=n_boot, seed=seed)
    out["bss_vs_clim"] = metrics.brier_skill_score(y, p, p_ref)
    out["log_loss"] = metrics.log_loss(y, p)
    for target, thr in thresholds.items():
        prec, rec = metrics.precision_recall_at(y, p, thr) if not np.isnan(thr) else (float("nan"), 0.0)
        tag = "" if target == MAIN_RECALL else f"_r{int(round(target * 100))}"
        out[f"precision{tag}"], out[f"recall{tag}"] = prec, rec
    return out


def views(frame):
    """Boolean masks for the two views."""
    return {"a_all": np.ones(len(frame), dtype=bool), "b_clear": frame["clear"].values}


def results_table(rows):
    return pd.DataFrame(rows)
