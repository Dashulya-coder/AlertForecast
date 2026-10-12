"""Phase 1: do other regions alert before Kyiv city? Train period only.

Two views of the same question, both computed strictly inside the train period:
  A. Lead time: for each Kyiv alert start s, the nearest start of region R within [s-6h, s+1h].
     lead = s - r_start (positive: R alerted first). Compared with a null made by moving Kyiv onsets
     back by several days, which keeps Kyiv's own clustering but breaks the alignment with R.
  B. Conditional rate: among origins where Kyiv is clear at t, P(Kyiv alert in next h) when R is
     active at t versus clear. Uses state at t only.
Nothing from the validation or test period is used (Kyiv starts are cut 1h before validation starts).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as c

BEFORE, AFTER = 6 * 3600, 1 * 3600
NULL_SHIFTS_DAYS = (3, 5, 7, 10)  # moved back in time only
LEADS_MIN = (5, 15, 30)

df = c.load_alerts()
df = df[df.started_at_s < c.VAL_START]  # train period only
kyiv = df[df.region == "Kyiv City"]
k_start, k_end = c.merged_intervals(kyiv.started_at_s.values, kyiv.finished_at_s.values)
regions = sorted(r for r in df.region.unique() if r != "Kyiv City")
r_iv = {r: c.merged_intervals(df[df.region == r].started_at_s.values, df[df.region == r].finished_at_s.values) for r in regions}
r_starts_raw = {r: np.sort(df[df.region == r].started_at_s.values) for r in regions}


def lead_stats(onsets, r):
    """Lead time (seconds) of region r for each onset, NaN where no start is within the window."""
    rs = r_starts_raw[r]
    lo = np.searchsorted(rs, onsets - BEFORE, side="left")
    hi = np.searchsorted(rs, onsets + AFTER, side="right")
    lead = np.full(len(onsets), np.nan)
    for i, (a, b, s) in enumerate(zip(lo, hi, onsets)):
        if b > a:
            cand = rs[a:b]
            lead[i] = s - cand[np.argmin(np.abs(cand - s))]
    return lead


def analyse(period_start, label):
    onsets = k_start[(k_start >= period_start) & (k_start < c.VAL_START - AFTER)]
    print(f"\n=== {label}: {len(onsets)} Kyiv alert starts ===")
    rows = []
    for r in regions:
        lead = lead_stats(onsets, r)
        has = ~np.isnan(lead)
        row = {"region": r, "has_start_in_window": has.mean()}
        for m in LEADS_MIN:
            row[f"lead>={m}m"] = np.mean(lead >= m * 60)  # NaN compares False
        if has.any():
            row["median_lead_min"] = np.median(lead[has]) / 60
        null = []
        for d in NULL_SHIFTS_DAYS:
            sh = onsets - d * 86400
            sh = sh[sh >= period_start + BEFORE]
            nl = lead_stats(sh, r)
            null.append([np.mean(nl >= 15 * 60), np.mean(~np.isnan(nl))])
        null = np.mean(null, axis=0)
        row["null_lead>=15m"] = null[0]
        row["excess_lead>=15m"] = row["lead>=15m"] - null[0]
        rows.append(row)
    out = pd.DataFrame(rows).set_index("region").sort_values("excess_lead>=15m", ascending=False)
    pd.set_option("display.width", 220)
    print(out.round(3).head(12).to_string())
    print("... Kyivska oblast row:")
    print(out.loc[["Kyivska oblast"]].round(3).to_string())
    return out


def active_for_at_least(rs, re, t, min_age_s):
    """R active at t AND its current alert started at least min_age_s before t (state known at t)."""
    idx = c.first_ending_after(rs, re, t)
    ok = idx < len(rs)
    out = np.zeros(len(t), dtype=bool)
    out[ok] = (t[ok] - rs[idx[ok]]) >= min_age_s
    return out


def conditional_rates(period_start, label, min_age_s=0):
    t = c.origins(period_start, c.VAL_START - 6 * 3600)  # labels need 6h of train-period data
    t = t[~c.active_at(k_start, k_end, t)]  # view (b): Kyiv clear at t
    base = {h: c.label(k_start, k_end, t, h).mean() for h in (1, 3)}
    print(f"\n=== {label}: P(Kyiv alert in next h | R active for >= {min_age_s // 60} min at t), Kyiv clear at t, {len(t)} origins ===")
    print(f"base rate: 1h {base[1]:.3f}  3h {base[3]:.3f}")
    rows = []
    for r in regions:
        rs, re = r_iv[r]
        act = active_for_at_least(rs, re, t, min_age_s)
        if act.sum() < 200:
            continue
        row = {"region": r, "n_active": int(act.sum()), "share_active": act.mean()}
        for h in (1, 3):
            y = c.label(k_start, k_end, t, h)
            row[f"p{h}h_active"] = y[act].mean()
            row[f"p{h}h_clear"] = y[~act].mean()
            row[f"lift{h}h"] = row[f"p{h}h_active"] / row[f"p{h}h_clear"]
        rows.append(row)
    out = pd.DataFrame(rows).set_index("region").sort_values("lift1h", ascending=False)
    print(out.round(3).head(8).to_string())
    print("... Kyivska oblast:")
    print(out.loc[["Kyivska oblast"]].round(3).to_string() if "Kyivska oblast" in out.index else "(too few active origins)")
    return out


def n_active_regions_table(period_start, label):
    """Generic 'wave is underway' baseline: how many OTHER regions are active at t, regardless of which."""
    t = c.origins(period_start, c.VAL_START - 6 * 3600)
    t = t[~c.active_at(k_start, k_end, t)]
    n_act = np.zeros(len(t), dtype=int)
    for r in regions:
        rs, re = r_iv[r]
        n_act += active_for_at_least(rs, re, t, 15 * 60)
    bins = pd.cut(n_act, [-1, 0, 2, 5, 10, 30], labels=["0", "1-2", "3-5", "6-10", "11+"])
    res = pd.DataFrame({"bin": bins, "y1": c.label(k_start, k_end, t, 1), "y3": c.label(k_start, k_end, t, 3)})
    print(f"\n=== {label}: P(Kyiv alert in next h | number of OTHER regions active for >=15 min at t), Kyiv clear ===")
    print(res.groupby("bin", observed=True).agg(n=("y1", "size"), p1h=("y1", "mean"), p3h=("y3", "mean")).round(3).to_string())


last12 = c.VAL_START - 365 * 86400
analyse(last12, "last 12 months of train")
analyse(int(pd.Timestamp("2022-03-01", tz="UTC").timestamp()), "full train")
conditional_rates(last12, "last 12 months of train", min_age_s=0)
conditional_rates(last12, "last 12 months of train", min_age_s=15 * 60)
n_active_regions_table(last12, "last 12 months of train")
