"""Phase 3, Stage C: do the neighbour-region features help? Three windows, same adoption rule as Stage A.

Base = the Stage A/B winner (stored predictions). Candidate = same recipe + nb_* feature group (47 columns).
Primary: log loss in view (b), mean over windows. Adoption: CI below 0 at at least 2 of 3 horizons, never
significantly worse on log loss, PR-AUC never significantly worse. The test split is not reachable from this script.
"""
import os
import pickle

import numpy as np
import pandas as pd

from alertforecast import config, metrics, model, protocol
from alertforecast.neighbors import neighbor_columns, neighbor_regions
from alertforecast.data import load_alerts

N_BOOT = 300
WINDOWS = ("val", "w1", "w2")
with open(os.path.join(config.ROOT, "data", "processed", "stage_a_preds.pkl"), "rb") as f:
    stage_a = pickle.load(f)

full = protocol.load_full_frame(neighbors=True)
regions = neighbor_regions(load_alerts())
base = model.stage_a_recipes()[0]
cand = model.Recipe("hlnone_base_nb", tuple(base.features) + tuple(neighbor_columns(regions)), None)
print(f"{len(regions)} neighbour regions, {len(cand.features) - len(base.features)} extra features", flush=True)

preds = {}
for h in config.HORIZONS_H:
    for w in WINDOWS:
        target, p = protocol.run_window(cand, full, w, h)
        preds[(cand.name, w, h)] = {"y": target[f"y{h}"].values, "p": p, "clear": target["clear"].values, "day": target["day"].values}
        print(f"fitted h={h} window={w}", flush=True)
with open(os.path.join(config.ROOT, "data", "processed", "stage_c_preds.pkl"), "wb") as f:
    pickle.dump(preds, f)


def get(name, w, h):
    return preds[(name, w, h)] if name == cand.name else stage_a[(name, w, h)]


def wins(name, h):
    out = []
    for w in WINDOWS:
        e = get(name, w, h)
        m = e["clear"]
        out.append((e["y"][m], e["p"][m], e["day"][m]))
    return out


rows = []
for h in config.HORIZONS_H:
    a, b = wins(cand.name, h), wins(base.name, h)
    ll = metrics.multi_window_diff_ci(metrics.log_loss, a, b, n_boot=N_BOOT)
    pr = metrics.multi_window_diff_ci(metrics.pr_auc, a, b, n_boot=N_BOOT)
    br = metrics.multi_window_diff_ci(metrics.brier, a, b, n_boot=N_BOOT)
    row = {"h": h, "dLogLoss": ll[0], "ll_lo": ll[1], "ll_hi": ll[2], "dPRAUC": pr[0], "pr_lo": pr[1], "pr_hi": pr[2],
           "dBrier": br[0], "br_lo": br[1], "br_hi": br[2]}
    for w, (y1, p1, _), (_, p0, _) in zip(WINDOWS, a, b):
        row[f"dLL_{w}"] = metrics.log_loss(y1, p1) - metrics.log_loss(y1, p0)
        row[f"dPR_{w}"] = metrics.pr_auc(y1, p1) - metrics.pr_auc(y1, p0)
    rows.append(row)
dd = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("\nNeighbour group vs base, view (b), mean over windows (negative log loss / positive PR-AUC = better):")
print(dd.round(4).to_string(index=False))

wins_n = int((dd.ll_hi < 0).sum())
worse = bool((dd.ll_lo > 0).any())
pr_worse = bool((dd.pr_hi < 0).any())
ok = wins_n >= 2 and not worse and not pr_worse
print(f"\nAdoption rule: horizons won {wins_n}, sig. worse log loss {worse}, sig. worse PR-AUC {pr_worse} -> "
      f"{'ADOPT the neighbour group' if ok else 'DO NOT adopt the neighbour group'}")

# context: mean prediction / observed rate and absolute PR-AUC in view (b)
ctx = []
for h in config.HORIZONS_H:
    for name in (base.name, cand.name):
        r = {"h": h, "recipe": name}
        for w in WINDOWS:
            e = get(name, w, h)
            m = e["clear"]
            r[f"pr_auc_{w}"] = metrics.pr_auc(e["y"][m], e["p"][m])
            r[f"p_over_rate_{w}"] = e["p"][m].mean() / e["y"][m].mean()
        ctx.append(r)
print("\nContext, view (b):")
print(pd.DataFrame(ctx).round(3).to_string(index=False))
dd.to_csv(os.path.join(config.ROOT, "reports", "phase3_stage_c_neighbors.csv"), index=False)
