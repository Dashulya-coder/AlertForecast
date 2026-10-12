"""Phase 3: final model choice, GBM vs the recent-activity logistic baseline. Rule fixed in CLAUDE.md before running.

Rule (user): use the logistic baseline UNLESS the GBM's log-loss CI is entirely below 0 at 2 of the 3 horizons.
Difference = GBM - logistic on view (b), mean over the windows val / w1 / w2, multi-window paired day-block bootstrap.
The test split is not reachable from this script.
"""
import os
import pickle

import numpy as np
import pandas as pd

from alertforecast import config, metrics, protocol, splits
from alertforecast.baselines import RecentActivity

N_BOOT = 300
WINDOWS = ("val", "w1", "w2")
with open(os.path.join(config.ROOT, "data", "processed", "stage_a_preds.pkl"), "rb") as f:
    gbm_preds = pickle.load(f)  # frozen recipe 'hlnone_base' (Stage A winner), fit on all purged rows
full = protocol.load_full_frame()

rows = []
for h in config.HORIZONS_H:
    gbm_b, log_b, gbm_a, log_a = [], [], [], []
    for w in WINDOWS:
        start = config.EVAL_WINDOWS[w][0]
        fit = protocol.rows(full, splits.fit_origins_before(start, h))
        tgt = protocol.rows(full, splits.window_origins(w))
        g = gbm_preds[("hlnone_base", w, h)]
        assert (g["y"] == tgt[f"y{h}"].values).all()  # same rows, same order
        # view (b): logistic fit on clear rows only
        lb = RecentActivity().fit(fit[fit.clear], fit[fit.clear][f"y{h}"].values)
        m = tgt.clear.values
        y_b = tgt[f"y{h}"].values[m]
        gbm_b.append((y_b, g["p"][m], tgt.day.values[m]))
        log_b.append((y_b, lb.predict(tgt[m]), tgt.day.values[m]))
        # view (a), for information: logistic fit on all rows
        la = RecentActivity().fit(fit, fit[f"y{h}"].values)
        gbm_a.append((tgt[f"y{h}"].values, g["p"], tgt.day.values))
        log_a.append((tgt[f"y{h}"].values, la.predict(tgt), tgt.day.values))
    row = {"h": h}
    for view, (a, b) in (("b", (gbm_b, log_b)), ("a", (gbm_a, log_a))):
        for fn_name, fn in (("logloss", metrics.log_loss), ("prauc", metrics.pr_auc), ("brier", metrics.brier)):
            d, lo, hi = metrics.multi_window_diff_ci(fn, a, b, n_boot=N_BOOT)
            row[f"{fn_name}_{view}_diff"], row[f"{fn_name}_{view}_lo"], row[f"{fn_name}_{view}_hi"] = d, lo, hi
            row[f"{fn_name}_{view}_gbm"] = float(np.mean([fn(y, p) for y, p, _ in a]))
            row[f"{fn_name}_{view}_logistic"] = float(np.mean([fn(y, p) for y, p, _ in b]))
    rows.append(row)

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("PRIMARY: log loss, view (b), GBM minus logistic (negative = GBM better), mean over windows")
print(res[["h", "logloss_b_gbm", "logloss_b_logistic", "logloss_b_diff", "logloss_b_lo", "logloss_b_hi"]].round(4).to_string(index=False))
print("\nFor information, view (b) PR-AUC and Brier, and view (a) log loss (GBM minus logistic):")
print(res[["h", "prauc_b_diff", "prauc_b_lo", "prauc_b_hi", "brier_b_diff", "brier_b_lo", "brier_b_hi",
           "logloss_a_diff", "logloss_a_lo", "logloss_a_hi"]].round(4).to_string(index=False))

n_win = int((res.logloss_b_hi < 0).sum())
choice = "gbm" if n_win >= 2 else "logistic"
print(f"\nHorizons where the GBM's log-loss CI is entirely below 0: {n_win} of 3 -> FINAL MODEL: "
      f"{'GBM (frozen Stage A-C recipe)' if choice == 'gbm' else 'recent-activity logistic baseline'}")
res["final_model"] = choice
res.to_csv(os.path.join(config.ROOT, "reports", "phase3_final_model_choice.csv"), index=False)
