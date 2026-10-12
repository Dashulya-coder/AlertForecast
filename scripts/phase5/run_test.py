"""Phase 5: the single evaluation of the frozen recipe.

    python scripts/phase5/run_test.py dry    # identical code path on the VALIDATION window (checks, no test data)
    python scripts/phase5/run_test.py test   # the real test split, runs ONCE (marker file reports/.test_run_done)

Frozen recipe (CLAUDE.md): GBM, base features, no recency weighting, no recent-rate group, no neighbour group, no
calibration, fixed hyperparameters. The final model is fit on all origins strictly purged before the evaluation start
(train + val for the test). Thresholds for recall 0.5 / 0.8 / 0.9 come from the pooled out-of-time predictions of the
three 28-day blocks before the evaluation start, per view. Nothing is tuned here.
"""
import datetime
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from alertforecast import config, evaluate, metrics, model, protocol, splits
from alertforecast.baselines import Climatology, Persistence, RecentActivity
from alertforecast.data import complete_until, load_alerts

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("dry", "test"):
    sys.exit("usage: run_test.py dry|test")
IS_TEST = MODE == "test"
PREFIX = "test_" if IS_TEST else "val_dry_"
OUT = os.path.join(config.ROOT, "reports")
FIG = os.path.join(OUT, "figures")
MARKER = os.path.join(OUT, ".test_run_done")
N_BOOT = 1000 if IS_TEST else 200
os.makedirs(FIG, exist_ok=True)

if IS_TEST:
    if os.path.exists(MARKER):
        sys.exit(f"The test split was already evaluated ({open(MARKER).read().strip()}). Not running it again.")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "tests"],
                                capture_output=True, text=True, cwd=config.ROOT).stdout.strip())
    import hashlib
    code = hashlib.sha256()
    for root in ("src", os.path.join("scripts", "phase5")):
        for dirpath, _, names in sorted(os.walk(os.path.join(config.ROOT, root))):
            for n in sorted(names):
                if n.endswith(".py"):
                    code.update(open(os.path.join(dirpath, n), "rb").read())
    with open(MARKER, "w") as f:  # written BEFORE any result is produced or printed
        f.write(json.dumps({"run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_commit": commit,
                            "uncommitted_code_changes": dirty, "code_sha256": code.hexdigest()}))

alerts = load_alerts()
cu = complete_until(alerts)
eval_name = "test" if IS_TEST else "val"
eval_start, eval_end = (config.TEST_START, config.TEST_END) if IS_TEST else (config.VAL_START, config.TEST_START)
full = protocol.load_full_frame(end_s=eval_end, allow_test=IS_TEST)
recipe = model.stage_a_recipes()[0]
assert recipe.name == "hlnone_base" and recipe.half_life_days is None
print(f"mode={MODE}  evaluation {pd.Timestamp(eval_start, unit='s', tz='UTC')} .. {pd.Timestamp(eval_end, unit='s', tz='UTC')}  "
      f"Kyiv complete until {pd.Timestamp(cu, unit='s', tz='UTC')}", flush=True)

kyiv_naive_starts = np.sort(alerts[(alerts.region == config.KYIV_REGION) & alerts.naive].started_at_s.values)
stage_a = None
if not IS_TEST:
    with open(os.path.join(config.ROOT, "data", "processed", "stage_a_preds.pkl"), "rb") as f:
        stage_a = pickle.load(f)

rows, diffs, weekly, calib, sens, meta = [], [], [], [], [], []
for h in config.HORIZONS_H:
    fit = protocol.rows(full, splits.fit_origins_before(eval_start, h))
    assert fit["t"].max() + h * config.HOUR_S < eval_start  # strict purge
    origins = splits.eval_origins(eval_name, allow_test=IS_TEST, horizon_h=h, complete_until=cu)
    ev = protocol.rows(full, origins)
    y_all = ev[f"y{h}"].values
    gbm = recipe.fit(fit, fit[f"y{h}"].values, eval_start)
    p_all = gbm.predict(ev)
    if not IS_TEST:  # the dry run must reproduce the Stage A predictions exactly
        assert np.allclose(p_all, stage_a[("hlnone_base", "val", h)]["p"], atol=1e-12), "dry run differs from Stage A"
    oot = protocol.oot_calibration_data_before(recipe, full, eval_start, h)
    thr = {"a_all": metrics.thresholds_for_recalls(oot.y.values, oot.p.values),
           "b_clear": metrics.thresholds_for_recalls(oot.y[oot.clear].values, oot.p[oot.clear].values)}
    meta.append({"h": h, "fit_rows": len(fit), "eval_origins": len(ev), "oot_rows": len(oot),
                 **{f"thr_{v}_{r}": t for v, d in thr.items() for r, t in d.items()}})
    print(f"h={h}h: fit rows {len(fit)}, evaluation origins {len(ev)}, OOT rows {len(oot)}", flush=True)

    for view in ("a_all", "b_clear"):
        m = np.ones(len(ev), dtype=bool) if view == "a_all" else ev["clear"].values
        v, y, p = ev[m], y_all[m], p_all[m]
        f = fit if view == "a_all" else fit[fit.clear]
        yf = f[f"y{h}"].values
        clim = Climatology(("hour",)).fit(f, yf)
        ref = clim.predict(v)
        preds = {"GBM (final)": p, "persistence": Persistence().predict(v), clim.name: ref}
        for b in (Climatology(("hour", "dow")), RecentActivity()):
            preds[b.name] = b.fit(f, yf).predict(v)
        preds["DIAGNOSTIC_const_eval_rate"] = np.full(len(v), y.mean())  # not deployable, not a baseline to beat
        for name, pp in preds.items():
            if name == "GBM (final)":
                thresholds = thr[view]
            elif name == "persistence":
                thresholds = {r: 0.5 for r in metrics.RECALL_TARGETS}
            else:
                thresholds = {r: float("nan") for r in metrics.RECALL_TARGETS}  # operating points: final model only
            r = evaluate.evaluate_view(v, y, pp, ref, thresholds, n_boot=N_BOOT)
            if name not in ("GBM (final)", "persistence"):
                for k in [k for k in r if k.startswith(("precision", "recall"))]:
                    r[k] = float("nan")
            rows.append({"horizon_h": h, "view": view, "model": name, **r})
        for other in ("climatology_hour", "recent_activity_24h", "DIAGNOSTIC_const_eval_rate"):
            for fn_name, fn in (("pr_auc", metrics.pr_auc), ("brier", metrics.brier), ("log_loss", metrics.log_loss)):
                d, lo, hi = metrics.block_bootstrap_diff_ci(fn, y, p, preds[other], v["day"].values, n_boot=N_BOOT)
                diffs.append({"horizon_h": h, "view": view, "vs": other, "metric": fn_name, "diff": d, "lo": lo, "hi": hi})
        # weekly calibration drift
        wk = ((v["t"].values - eval_start) // (7 * config.DAY_S)).astype(int)
        for k in np.unique(wk):
            mk = wk == k
            weekly.append({"horizon_h": h, "view": view, "week": int(k) + 1, "n": int(mk.sum()), "base_rate": y[mk].mean(),
                           "mean_p": p[mk].mean(), "p_over_rate": p[mk].mean() / y[mk].mean() if y[mk].mean() > 0 else np.nan,
                           "brier": metrics.brier(y[mk], p[mk])})
        ct = metrics.calibration_table(y, p)
        ct.insert(0, "view", view)
        ct.insert(0, "horizon_h", h)
        calib.append(ct)
        # sensitivity: drop origins whose label window (t, t+h] touches a naive alert's imputed 30-minute span
        t_ = v["t"].values
        touch = (np.searchsorted(kyiv_naive_starts, t_ + h * config.HOUR_S, side="right")
                 - np.searchsorted(kyiv_naive_starts, t_ - 1800, side="right")) > 0
        keep = ~touch
        sens.append({"horizon_h": h, "view": view, "n_all": len(y), "n_kept": int(keep.sum()),
                     "base_rate_all": y.mean(), "base_rate_kept": y[keep].mean(),
                     "pr_auc_all": metrics.pr_auc(y, p), "pr_auc_kept": metrics.pr_auc(y[keep], p[keep]),
                     "brier_all": metrics.brier(y, p), "brier_kept": metrics.brier(y[keep], p[keep]),
                     "log_loss_all": metrics.log_loss(y, p), "log_loss_kept": metrics.log_loss(y[keep], p[keep])})

res, dd, wk, sn, mt = (pd.DataFrame(x) for x in (rows, diffs, weekly, sens, meta))
ct = pd.concat(calib, ignore_index=True)
for name, df in (("results", res), ("paired_diffs", dd), ("weekly", wk), ("calibration", ct), ("naive_sensitivity", sn), ("setup", mt)):
    df.to_csv(os.path.join(OUT, f"{PREFIX}{name}.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
show = ["model", "base_rate", "pr_auc", "pr_auc_lo", "pr_auc_hi", "brier", "bss_vs_clim", "log_loss",
        "precision", "recall", "precision_r50", "recall_r50", "precision_r90", "recall_r90"]
for (h, view), g in res.groupby(["horizon_h", "view"]):
    print(f"\n--- {h}h, view {view}, n={g.n.iloc[0]} ---")
    print(g[show].round(3).to_string(index=False))
print("\nPaired differences, GBM minus other (PR-AUC: + better; Brier / log loss: - better):")
print(dd.round(4).to_string(index=False))
print("\nWeekly calibration drift:")
print(wk.round(3).to_string(index=False))
print("\nSensitivity, excluding origins whose label window touches a naive alert:")
print(sn.round(3).to_string(index=False))

# ---------------- figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {1: ("#2a78d6", "o"), 3: ("#eb6834", "s"), 6: ("#1baf7a", "D")}  # validated slots 1-3, distinct markers
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10})
label = "test" if IS_TEST else "validation (dry run)"

# 1. reliability diagrams: rows = view (b headline, then a), columns = horizon
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
for i, view in enumerate(("b_clear", "a_all")):
    for j, h in enumerate(config.HORIZONS_H):
        ax = axes[i, j]
        g = ct[(ct.horizon_h == h) & (ct.view == view)]
        top = float(max(g.p.max(), g.y.max())) * 1.08
        ax.plot([0, top], [0, top], color=INK2, lw=1, ls=(0, (4, 3)))
        c, mk = SERIES[h]
        ax.plot(g.p, g.y, color=c, lw=2, marker=mk, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{h}h ahead, " + ("Kyiv clear at t (headline)" if view == "b_clear" else "all origins"), fontsize=10, loc="left", color=INK)
        if j == 0:
            ax.set_ylabel("observed alert rate")
        if i == 1:
            ax.set_xlabel("mean predicted probability")
        ax.text(0.97, 0.05, "perfect calibration", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
fig.suptitle(f"Reliability on {label}: equal-count bins, 10 per panel", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"{PREFIX}reliability.png"), dpi=150)
plt.close(fig)

# 2. weekly drift (view b): observed rate and mean prediction / observed rate, per horizon
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
wb = wk[wk.view == "b_clear"]
for h in config.HORIZONS_H:
    g = wb[wb.horizon_h == h]
    c, mk = SERIES[h]
    for ax, col in ((axes[0], "base_rate"), (axes[1], "p_over_rate")):
        ax.plot(g.week, g[col], color=c, lw=2, marker=mk, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{h}h")
        ax.annotate(f"{h}h", (g.week.iloc[-1], g[col].iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
axes[0].set_title("Observed alert rate per week (Kyiv clear at t)", fontsize=10, loc="left")
axes[1].set_title("Mean predicted / observed rate per week", fontsize=10, loc="left")
axes[1].axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
axes[1].text(0.6, 1.0, "calibrated", fontsize=8, color=INK2, va="bottom")
for ax in axes:
    ax.set_xlabel("week of the evaluation window")
    ax.set_xticks(sorted(wb.week.unique()))
    ax.set_xlim(0.6, wb.week.max() + 0.7)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")  # the right panel is direct-labelled
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"{PREFIX}weekly_drift.png"), dpi=150)
plt.close(fig)
print(f"\nsaved {PREFIX}*.csv in reports/ and figures in reports/figures/")
