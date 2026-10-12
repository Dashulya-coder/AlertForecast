"""How much does the naive-flag leak change the picture? Validation and train only; test is not touched.

Compares the clear-at-t state built from the dataset's ends (leaky: naive alerts end at start+30min) with the
real-time state (naive alerts stay open until the next activation).
"""
import numpy as np
import pandas as pd

from alertforecast import config, splits
from alertforecast.data import Intervals, load_alerts
from alertforecast.frame import build_frame

alerts = load_alerts()
label_iv = Intervals.for_region(alerts)
state_iv = Intervals.for_region_state(alerts)

for name, t in (("train", splits.grid(config.TRAIN_ORIGIN_START, config.VAL_START)),
                ("train, last 12 months", splits.grid(config.VAL_START - 365 * 86400, config.VAL_START)),
                ("val", splits.eval_origins("val"))):
    leaky = build_frame(t, label_iv, label_iv)  # dataset ends used as the state: the leaky version
    live = build_frame(t, state_iv, label_iv)
    shadow = leaky.clear.values & ~live.clear.values  # clear in the dataset, still open in real time
    print(f"\n=== {name}: {len(t)} origins ===")
    print(f"clear share: dataset-ends {leaky.clear.mean():.3f}, real-time {live.clear.mean():.3f}; "
          f"origins in the naive shadow: {shadow.sum()} ({shadow.mean():.1%} of all, {shadow.sum() / leaky.clear.sum():.1%} of dataset-clear)")
    other_clear = leaky.clear.values & live.clear.values
    for h in config.HORIZONS_H:
        y = live[f"y{h}"].values
        print(f"  h={h}h  label rate: shadow {y[shadow].mean():.3f} | other clear {y[other_clear].mean():.3f} | "
              f"dataset-clear view (b) {y[leaky.clear.values].mean():.3f} -> real-time view (b) {y[live.clear.values].mean():.3f}")

# The rule says no alert can start within 3h of a naive alert's start, so labels there are determined
t = splits.grid(config.TRAIN_ORIGIN_START, config.VAL_START)
naive = pd.DataFrame({"s": label_iv.starts, "e": label_iv.ends})
nv = alerts[(alerts.region == config.KYIV_REGION) & alerts.naive].started_at_s.values
viol = 0
checked = 0
for s in nv:
    for h in config.HORIZONS_H:
        tt = t[(t >= s + 1800) & (t + h * 3600 < s + 3 * 3600)]  # inside the shadow, window ends before s + 3h
        if len(tt):
            lab = np.array([label_iv.starts[np.searchsorted(label_iv.ends, x, side="right")] <= x + h * 3600 for x in tt])
            checked += len(tt)
            viol += int(lab.sum())
print(f"\nGuaranteed negatives: {checked} train origins inside a naive shadow whose window ends before start+3h; labels equal to 1: {viol}")
