"""Phase 1: per-horizon base rates in both views, train and validation only.

Test origins are deliberately never built in this script, so test labels are not looked at.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as c

df = c.load_alerts()
k = df[df.region == "Kyiv City"]
starts, ends = c.merged_intervals(k.started_at_s.values, k.finished_at_s.values)
print(f"Kyiv City alerts: {len(k)} raw, {len(starts)} after merging overlaps")

t = c.origins(c.TRAIN_ORIGIN_START, c.TEST_START)  # train + val origins only, nothing from test
active = c.active_at(starts, ends, t)
res = pd.DataFrame({"t": t, "clear": ~active})
for h in c.HORIZONS_H:
    res[f"y{h}"] = c.label(starts, ends, t, h)

dt = pd.to_datetime(res.t, unit="s", utc=True)
res["split"] = np.where(res.t >= c.VAL_START, "val", "train")
res["period"] = dt.dt.strftime("%Y")
res.loc[(res.t >= c.TRAIN_ORIGIN_START) & (dt >= pd.Timestamp("2026-01-01", tz="UTC")) & (res.split == "train"), "period"] = dt.dt.strftime("%Y-%m")
res.loc[res.split == "val", "period"] = "VAL"


def summarize(g):
    row = {"n": len(g), "clear": g.clear.mean()}
    b = g[g.clear]
    for h in c.HORIZONS_H:
        row[f"a_{h}h"] = g[f"y{h}"].mean()
    row["n_b"] = len(b)
    for h in c.HORIZONS_H:
        row[f"b_{h}h"] = b[f"y{h}"].mean()
    return pd.Series(row)


pd.set_option("display.width", 200)
tab = res.groupby("period").apply(summarize, include_groups=False)
print("\nBase rates (a = all origins, b = Kyiv clear at t), 30-min origins, grouped by UTC period.")
print("train: yearly for 2022-2025, monthly for 2026; VAL = 2026-08-17..2026-09-13 Kyiv days")
print(tab.round(3).to_string())
print("\nTrain overall:\n", summarize(res[res.split == "train"]).round(3).to_string())
print("\nValidation overall:\n", summarize(res[res.split == "val"]).round(3).to_string())

# last 12 months of train, used as the reference regime for the lead-time analysis
last12 = res[(res.split == "train") & (res.t >= c.VAL_START - 365 * 86400)]
print("\nLast 12 months of train:\n", summarize(last12).round(3).to_string())
