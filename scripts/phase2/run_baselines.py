"""Phase 2: fit baselines on purged training rows, score them on VALIDATION in both views.

The test split is never touched here (splits.eval_origins('test') is locked).
Features and the clear-at-t state come from the real-time view of the alerts (naive alerts stay open until the next
activation); labels come from the dataset's ends. Each view is its own task: view (b) baselines are fit on clear-at-t
training rows only, and Brier skill is against a climatology fitted for the same view.

Precision/recall are reported at recall targets 0.5, 0.8 (main) and 0.9, with thresholds chosen on validation itself.
That is in-sample on val (val is the tuning split); on test the thresholds must be frozen from val.
"""
import os

import numpy as np
import pandas as pd

from alertforecast import config, evaluate, metrics, splits
from alertforecast.baselines import Climatology, all_baselines
from alertforecast.data import Intervals, load_alerts
from alertforecast.frame import build_frame

N_BOOT = 500

alerts = load_alerts()
label_iv = Intervals.for_region(alerts)
state_iv = Intervals.for_region_state(alerts)
val = build_frame(splits.eval_origins("val"), state_iv, label_iv)

rows, diffs = [], []
for h in config.HORIZONS_H:
    fit = build_frame(splits.fit_origins("val", h), state_iv, label_iv)
    y_col = f"y{h}"
    for view in ("a_all", "b_clear"):
        f = fit if view == "a_all" else fit[fit.clear]
        v = val if view == "a_all" else val[val.clear]
        y_fit, y_val = f[y_col].values, v[y_col].values
        ref = Climatology(("hour",)).fit(f, y_fit).predict(v)
        preds = {}
        for model in all_baselines():
            model.fit(f, y_fit)
            preds[model.name] = model.predict(v)
        # Diagnostic only, NOT deployable and NOT a baseline to beat: always predicts validation's own base rate.
        # It shows how much of the Brier score is lost to the regime shift (train rate != val rate).
        preds["DIAGNOSTIC_const_val_rate"] = np.full(len(v), y_val.mean())
        for name, p in preds.items():
            if name == "persistence":  # hard 0/1 classifier: one operating point, no recall targets
                thr = {r: 0.5 for r in metrics.RECALL_TARGETS}
            else:
                thr = metrics.thresholds_for_recalls(y_val, p)
            r = evaluate.evaluate_view(v, y_val, p, ref, thr, n_boot=N_BOOT)
            rows.append({"horizon_h": h, "view": view, "model": name, "fit_rows": len(f), **r})
        # Paired differences against hour-of-day climatology on the same resampled days
        for name in ("climatology_hour_dow", "recent_activity_24h"):
            for fn_name, fn in (("pr_auc", metrics.pr_auc), ("brier", metrics.brier)):
                d, lo, hi = metrics.block_bootstrap_diff_ci(fn, y_val, preds[name], preds["climatology_hour"], v["day"].values, n_boot=N_BOOT)
                diffs.append({"horizon_h": h, "view": view, "model": name, "metric": fn_name, "diff_vs_clim_hour": d, "lo": lo, "hi": hi})

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
res["pr_auc_ci"] = res.apply(lambda r: f"{r.pr_auc:.3f} [{r.pr_auc_lo:.3f},{r.pr_auc_hi:.3f}]", axis=1)
res["brier_ci"] = res.apply(lambda r: f"{r.brier:.3f} [{r.brier_lo:.3f},{r.brier_hi:.3f}]", axis=1)
cols = ["model", "base_rate", "pr_auc_ci", "brier_ci", "bss_vs_clim", "log_loss", "precision", "recall",
        "precision_r50", "recall_r50", "precision_r90", "recall_r90"]
print(f"VALIDATION {pd.Timestamp(config.VAL_START, unit='s', tz='UTC')} .. {pd.Timestamp(config.TEST_START, unit='s', tz='UTC')}")
print("precision/recall = main point (target recall 0.8); _r50 / _r90 = targets 0.5 / 0.9; recall columns are the recall actually achieved.")
print(f"thresholds chosen on val (in-sample); persistence is a hard 0/1 classifier; CIs: day-block bootstrap, {N_BOOT} resamples")
for (h, view), g in res.groupby(["horizon_h", "view"]):
    print(f"\n--- horizon {h}h, view {view}, n={g.n.iloc[0]}, fit rows={g.fit_rows.iloc[0]} ---")
    print(g[cols].round(3).to_string(index=False))

dd = pd.DataFrame(diffs)
print("\n=== Paired differences vs hour-of-day climatology (same resampled days; PR-AUC: + is better, Brier: - is better) ===")
print(dd.round(3).to_string(index=False))

out_dir = os.path.join(config.ROOT, "reports")
os.makedirs(out_dir, exist_ok=True)
res.drop(columns=["pr_auc_ci", "brier_ci"]).to_csv(os.path.join(out_dir, "phase2_baselines_val.csv"), index=False)
dd.to_csv(os.path.join(out_dir, "phase2_baselines_val_paired_diffs.csv"), index=False)
print("\nsaved reports/phase2_baselines_val.csv and reports/phase2_baselines_val_paired_diffs.csv")
