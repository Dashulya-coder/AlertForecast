"""Phase 3, Stage A: recency weighting x recent-rate features, compared on three 28-day windows (val, w1, w2).

The rule below is the pre-registered one from CLAUDE.md. Primary: log loss in view (b), mean over windows.
Secondary: PR-AUC in view (b). Differences use the multi-window paired day-block bootstrap vs the base configuration.
The test split is not reachable from this script.
"""
import os
import pickle

import numpy as np
import pandas as pd

from alertforecast import config, metrics, model, protocol
from alertforecast.baselines import Climatology

N_BOOT = 300
WINDOWS = ("val", "w1", "w2")

full = protocol.load_full_frame()
recipes = model.stage_a_recipes()
base_name = recipes[0].name

preds = {}  # (recipe, window, horizon) -> dict of arrays
for h in config.HORIZONS_H:
    for window in WINDOWS:
        for recipe in recipes:
            target, p = protocol.run_window(recipe, full, window, h)
            preds[(recipe.name, window, h)] = {
                "y": target[f"y{h}"].values, "p": p, "clear": target["clear"].values, "day": target["day"].values}
        print(f"fitted h={h} window={window}", flush=True)

os.makedirs(os.path.join(config.ROOT, "data", "processed"), exist_ok=True)
with open(os.path.join(config.ROOT, "data", "processed", "stage_a_preds.pkl"), "wb") as f:
    pickle.dump(preds, f)


def view(entry, which):
    m = entry["clear"] if which == "b" else np.ones(len(entry["y"]), dtype=bool)
    return entry["y"][m], entry["p"][m], entry["day"][m]


def windows_of(name, h, which):
    return [view(preds[(name, w, h)], which) for w in WINDOWS]


def mean_metric(name, h, which, fn):
    return float(np.mean([fn(y, p) for y, p, _ in windows_of(name, h, which)]))


# ---- context: references fitted for view (b) on each window's purged fit rows ----
ref_rows = []
for h in config.HORIZONS_H:
    ll_clim, ll_const = [], []
    for w in WINDOWS:
        start = config.EVAL_WINDOWS[w][0]
        from alertforecast import splits
        fit = protocol.rows(full, splits.fit_origins_before(start, h))
        fit = fit[fit.clear]
        e = preds[(base_name, w, h)]
        tgt = protocol.rows(full, splits.window_origins(w))
        tgt = tgt[tgt.clear]
        clim = Climatology(("hour",)).fit(fit, fit[f"y{h}"].values).predict(tgt)
        ll_clim.append(metrics.log_loss(tgt[f"y{h}"].values, clim))
        ll_const.append(metrics.log_loss(tgt[f"y{h}"].values, np.full(len(tgt), tgt[f"y{h}"].mean())))
    ref_rows.append({"h": h, "climatology_hour": np.mean(ll_clim), "DIAGNOSTIC_const_window_rate": np.mean(ll_const)})
print("\nContext, view (b) mean log loss over windows:")
print(pd.DataFrame(ref_rows).round(4).to_string(index=False))

# ---- per-config table and paired differences vs base ----
rows, diffs = [], []
for h in config.HORIZONS_H:
    for r in recipes:
        row = {"h": h, "config": r.name}
        for which in ("a", "b"):
            row[f"logloss_{which}"] = mean_metric(r.name, h, which, metrics.log_loss)
            row[f"prauc_{which}"] = mean_metric(r.name, h, which, metrics.pr_auc)
            row[f"brier_{which}"] = mean_metric(r.name, h, which, metrics.brier)
        for w in WINDOWS:
            y, p, _ = view(preds[(r.name, w, h)], "b")
            row[f"logloss_b_{w}"] = metrics.log_loss(y, p)
        rows.append(row)
        if r.name == base_name:
            continue
        d_ll = metrics.multi_window_diff_ci(metrics.log_loss, windows_of(r.name, h, "b"), windows_of(base_name, h, "b"), n_boot=N_BOOT)
        d_pr = metrics.multi_window_diff_ci(metrics.pr_auc, windows_of(r.name, h, "b"), windows_of(base_name, h, "b"), n_boot=N_BOOT)
        diffs.append({"h": h, "config": r.name, "dLogLoss": d_ll[0], "ll_lo": d_ll[1], "ll_hi": d_ll[2],
                      "dPRAUC": d_pr[0], "pr_lo": d_pr[1], "pr_hi": d_pr[2]})

res = pd.DataFrame(rows)
dd = pd.DataFrame(diffs)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("\nMean over the 3 windows (view b unless marked a):")
print(res[["h", "config", "logloss_b", "prauc_b", "brier_b", "logloss_a", "prauc_a", "logloss_b_val", "logloss_b_w1", "logloss_b_w2"]].round(4).to_string(index=False))
print("\nPaired differences vs base (negative log loss diff = better; positive PR-AUC diff = better):")
print(dd.round(4).to_string(index=False))

# ---- pre-registered adoption rule ----
adopt = []
for name in dd.config.unique():
    g = dd[dd.config == name]
    wins = int((g.ll_hi < 0).sum())  # horizons where the primary CI is entirely below 0
    worse = bool((g.ll_lo > 0).any())  # significantly worse on the primary at any horizon
    pr_worse = bool((g.pr_hi < 0).any())  # significantly worse on PR-AUC at any horizon
    ok = wins >= 2 and not worse and not pr_worse
    mean_primary = float(res[res.config == name].logloss_b.mean())
    adopt.append({"config": name, "horizons_won": wins, "sig_worse_primary": worse, "sig_worse_prauc": pr_worse,
                  "qualifies": ok, "mean_logloss_b": mean_primary})
ad = pd.DataFrame(adopt)
print("\nAdoption rule:")
print(ad.round(4).to_string(index=False))
base_ll = float(res[res.config == base_name].logloss_b.mean())
q = ad[ad.qualifies]
if len(q):
    best = q.sort_values("mean_logloss_b").iloc[0]
    print(f"\nWINNER: {best.config} (mean log loss b {best.mean_logloss_b:.4f} vs base {base_ll:.4f})")
else:
    print(f"\nNo configuration qualifies. Keep the base: {base_name} (mean log loss b {base_ll:.4f})")

out = os.path.join(config.ROOT, "reports")
os.makedirs(out, exist_ok=True)
res.to_csv(os.path.join(out, "phase3_stage_a_configs.csv"), index=False)
dd.to_csv(os.path.join(out, "phase3_stage_a_diffs.csv"), index=False)
ad.to_csv(os.path.join(out, "phase3_stage_a_adoption.csv"), index=False)
