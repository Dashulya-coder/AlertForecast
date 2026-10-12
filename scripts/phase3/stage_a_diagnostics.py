"""Diagnostics for Stage A (read-only; uses the stored predictions). Not part of the adoption rule.

1. Calibration in the large: mean predicted probability / observed rate in view (b), per config and window.
   Tests the idea that recency weighting hurts because the model follows the previous, quieter regime.
2. Does the base GBM beat the simple baselines? PR-AUC and log loss in view (b), per window.
"""
import os
import pickle

import numpy as np
import pandas as pd

from alertforecast import config, metrics, protocol, splits
from alertforecast.baselines import Climatology, RecentActivity

WINDOWS = ("val", "w1", "w2")
with open(os.path.join(config.ROOT, "data", "processed", "stage_a_preds.pkl"), "rb") as f:
    preds = pickle.load(f)
full = protocol.load_full_frame()
pd.set_option("display.width", 220)

print("1. Mean predicted / observed rate, view (b). 1.0 = right level; below 1 = under-predicting the window's rate.")
rows = []
for h in config.HORIZONS_H:
    for name in ("hlnone_base", "hl180_base", "hl60_base", "hl30_base"):
        row = {"h": h, "config": name}
        for w in WINDOWS:
            e = preds[(name, w, h)]
            m = e["clear"]
            row[w] = e["p"][m].mean() / e["y"][m].mean()
        rows.append(row)
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n2. Base GBM vs simple baselines, view (b), per window (baselines fit on clear fit rows only).")
rows = []
for h in config.HORIZONS_H:
    for w in WINDOWS:
        start = config.EVAL_WINDOWS[w][0]
        fit = protocol.rows(full, splits.fit_origins_before(start, h))
        fit = fit[fit.clear]
        tgt = protocol.rows(full, splits.window_origins(w))
        tgt = tgt[tgt.clear]
        y = tgt[f"y{h}"].values
        gbm = preds[("hlnone_base", w, h)]
        gp = gbm["p"][gbm["clear"]]
        clim = Climatology(("hour",)).fit(fit, fit[f"y{h}"].values).predict(tgt)
        rec = RecentActivity().fit(fit, fit[f"y{h}"].values).predict(tgt)
        rows.append({"h": h, "window": w, "base_rate": y.mean(),
                     "prauc_gbm": metrics.pr_auc(y, gp), "prauc_recent": metrics.pr_auc(y, rec), "prauc_clim": metrics.pr_auc(y, clim),
                     "ll_gbm": metrics.log_loss(y, gp), "ll_recent": metrics.log_loss(y, rec), "ll_clim": metrics.log_loss(y, clim),
                     "ll_const_window_rate": metrics.log_loss(y, np.full(len(y), y.mean()))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
