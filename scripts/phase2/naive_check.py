"""Check the claim: `naive=True` (end = start + 30 min) is only set when the next alert in the same region starts
more than 3 hours after this alert's start, so a 30-minute duration reveals the future.

Rule in the dataset's processor (volunteer_etryvoga_processor.py, process_message): on a new activation message for a
region that still has an open alert (no end message seen):
  - if the new message is > 3h after the open alert's start: the open alert is written with
    finished_at = started_at + 30 min and naive = True;
  - otherwise the open alert is silently overwritten (dropped, never written).
Open alerts are only written to the CSV when they close, so alerts without an end message appear in the file only
once the next activation arrives.
"""
import numpy as np
import pandas as pd

from alertforecast import config
from alertforecast.data import load_alerts

al = load_alerts()
al = al.sort_values(["region", "started_at_s"]).reset_index(drop=True)
al["next_start_s"] = al.groupby("region").started_at_s.shift(-1)
al["gap_to_next_start_h"] = (al.next_start_s - al.started_at_s) / 3600
al["dur_s"] = al.finished_at_s - al.started_at_s
H3 = 3.0


def report(df, name):
    nv, ok = df[df.naive], df[~df.naive]
    nv_has_next = nv.dropna(subset=["gap_to_next_start_h"])
    print(f"\n=== {name}: {len(df)} alerts, {len(nv)} naive ({len(nv) / len(df):.1%}) ===")
    print(f"naive duration is always exactly 1800s: {(nv.dur_s == 1800).all()}  (non-naive with duration exactly 1800s: {(ok.dur_s == 1800).sum()} of {len(ok)})")
    print(f"naive alerts with a next alert in the file: {len(nv_has_next)}; without: {len(nv) - len(nv_has_next)}")
    g = nv_has_next.gap_to_next_start_h
    print(f"naive: gap from this START to next START, hours: min {g.min():.3f}, p5 {g.quantile(.05):.2f}, median {g.median():.2f}")
    print(f"naive with gap <= 3h: {(g <= H3).sum()} of {len(g)}  <- the claim predicts 0")
    g2 = ok.dropna(subset=["gap_to_next_start_h"]).gap_to_next_start_h
    print(f"non-naive: share with gap <= 3h: {(g2 <= H3).mean():.1%}, min {g2.min():.3f}h")


report(al, "all regions")
kyiv = al[al.region == config.KYIV_REGION]
report(kyiv, "Kyiv City")
kyiv_recent = kyiv[kyiv.started_at_s >= config.VAL_START - 365 * 86400]
print(f"\nKyiv, last 12 months of train + val + test period: naive share {kyiv_recent.naive.mean():.1%}")
