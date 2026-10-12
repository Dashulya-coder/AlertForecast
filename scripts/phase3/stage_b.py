"""Phase 3, Stage B: calibration candidates on the Stage A winner (the base recipe), three windows.

For each window and horizon: K=3 out-of-time calibration blocks before the window (each block predicted by a model fit
strictly before it), calibrator fit on the pooled blocks, applied to the window's own predictions (the Stage A
predictions of the same recipe). Primary: Brier in view (b), mean over windows. Guard: log loss not significantly worse.
PR-AUC cannot change under a monotone calibrator; this is asserted.
Thresholds for recall 0.5/0.8/0.9 are chosen on the pooled out-of-time predictions and applied to each window.
The test split is not reachable from this script.
"""
import os
import pickle

import numpy as np
import pandas as pd

from alertforecast import calibration, config, metrics, model, protocol

N_BOOT = 300
WINDOWS = ("val", "w1", "w2")
recipe = model.stage_a_recipes()[0]  # base recipe: no recency weighting, base features (Stage A outcome)
assert recipe.name == "hlnone_base"

with open(os.path.join(config.ROOT, "data", "processed", "stage_a_preds.pkl"), "rb") as f:
    stage_a = pickle.load(f)
full = protocol.load_full_frame()

cands = {c.name: c for c in calibration.stage_b_calibrators()}
out = {}  # (calibrator, window, h) -> dict
oot_store = {}
for h in config.HORIZONS_H:
    for w in WINDOWS:
        oot = protocol.oot_calibration_data(recipe, full, w, h)
        oot_store[(w, h)] = oot
        win = stage_a[(recipe.name, w, h)]
        block_w = np.array(calibration.BLOCK_WEIGHTS)[oot.block.values]
        for name, cal in cands.items():
            wts = block_w if name == "platt_recency" else None
            cal.fit(oot.p.values, oot.y.values, weights=wts)
            pc = cal.predict(win["p"])
            assert abs(metrics.pr_auc(win["y"], pc) - metrics.pr_auc(win["y"], win["p"])) < 1e-9 or name == "none"
            out[(name, w, h)] = {"y": win["y"], "p": pc, "clear": win["clear"], "day": win["day"]}
        print(f"calibrated h={h} window={w} (OOT rows {len(oot)})", flush=True)
with open(os.path.join(config.ROOT, "data", "processed", "stage_b_oot.pkl"), "wb") as f:
    pickle.dump(oot_store, f)


def view_b(e):
    m = e["clear"]
    return e["y"][m], e["p"][m], e["day"][m]


def wins(name, h):
    return [view_b(out[(name, w, h)]) for w in WINDOWS]


rows, diffs = [], []
for h in config.HORIZONS_H:
    for name in cands:
        row = {"h": h, "calibrator": name}
        for fn_name, fn in (("brier", metrics.brier), ("logloss", metrics.log_loss)):
            row[f"{fn_name}_b"] = np.mean([fn(y, p) for y, p, _ in wins(name, h)])
        for w in WINDOWS:
            y, p, _ = view_b(out[(name, w, h)])
            row[f"brier_b_{w}"] = metrics.brier(y, p)
            row[f"mean_p_over_rate_{w}"] = p.mean() / y.mean()
        rows.append(row)
        if name == "none":
            continue
        d_b = metrics.multi_window_diff_ci(metrics.brier, wins(name, h), wins("none", h), n_boot=N_BOOT)
        d_l = metrics.multi_window_diff_ci(metrics.log_loss, wins(name, h), wins("none", h), n_boot=N_BOOT)
        diffs.append({"h": h, "calibrator": name, "dBrier": d_b[0], "b_lo": d_b[1], "b_hi": d_b[2],
                      "dLogLoss": d_l[0], "l_lo": d_l[1], "l_hi": d_l[2]})
res, dd = pd.DataFrame(rows), pd.DataFrame(diffs)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("\nView (b), mean over windows; mean_p_over_rate = calibration in the large (1.0 = right level):")
print(res.round(4).to_string(index=False))
print("\nPaired differences vs no calibration (negative = better):")
print(dd.round(4).to_string(index=False))

adopt = []
for name in dd.calibrator.unique():
    g = dd[dd.calibrator == name]
    n_win = int((g.b_hi < 0).sum())
    brier_worse = bool((g.b_lo > 0).any())
    ll_worse = bool((g.l_lo > 0).any())
    adopt.append({"calibrator": name, "horizons_won": n_win, "sig_worse_brier": brier_worse, "sig_worse_logloss": ll_worse,
                  "qualifies": n_win >= 2 and not brier_worse and not ll_worse,
                  "mean_brier_b": float(res[res.calibrator == name].brier_b.mean())})
ad = pd.DataFrame(adopt)
print("\nAdoption rule:")
print(ad.round(4).to_string(index=False))
q = ad[ad.qualifies]
none_b = float(res[res.calibrator == "none"].brier_b.mean())
if len(q):
    best = q.sort_values("mean_brier_b").iloc[0]
    print(f"\nWINNER: {best.calibrator} (mean Brier b {best.mean_brier_b:.4f} vs none {none_b:.4f})")
else:
    print(f"\nNo calibrator qualifies. Keep none (mean Brier b {none_b:.4f})")

# ---- operating points: thresholds from the pooled out-of-time predictions, applied to each window ----
rows = []
for h in config.HORIZONS_H:
    for w in WINDOWS:
        oot = oot_store[(w, h)]
        win = out[("none", w, h)]
        for which in ("a", "b"):
            om = oot if which == "a" else oot[oot.clear]
            wm = np.ones(len(win["y"]), dtype=bool) if which == "a" else win["clear"]
            thr = metrics.thresholds_for_recalls(om.y.values, om.p.values)
            for target, t in thr.items():
                prec, rec = metrics.precision_recall_at(win["y"][wm], win["p"][wm], t)
                rows.append({"h": h, "window": w, "view": which, "target_recall": target, "threshold": t,
                             "precision": prec, "recall": rec, "base_rate": win["y"][wm].mean()})
op = pd.DataFrame(rows)
print("\nOperating points (threshold from the 3 pooled out-of-time blocks, applied to the window), mean over windows:")
print(op.groupby(["h", "view", "target_recall"])[["threshold", "precision", "recall", "base_rate"]].mean().round(3).to_string())
print("\nRecall achieved at target 0.8, per window (view b):")
print(op[(op.view == "b") & (op.target_recall == 0.8)][["h", "window", "recall", "precision", "base_rate"]].round(3).to_string(index=False))

o = os.path.join(config.ROOT, "reports")
res.to_csv(os.path.join(o, "phase3_stage_b_calibrators.csv"), index=False)
dd.to_csv(os.path.join(o, "phase3_stage_b_diffs.csv"), index=False)
ad.to_csv(os.path.join(o, "phase3_stage_b_adoption.csv"), index=False)
op.to_csv(os.path.join(o, "phase3_stage_b_operating_points.csv"), index=False)
