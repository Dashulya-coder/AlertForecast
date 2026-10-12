# AlertForecast

Mini pet-project: probabilistic forecasting of air-raid alerts in **Kyiv city** (not Kyiv oblast).
Everything here is judged by one rule: **no information from after the forecast time may be used for training or input.**

## Task

For each forecast origin `t`, output the probability that an alert is active in Kyiv city at any moment in `(t, t+h]`, for `h` in **{1h, 3h, 6h}**. There is no 24h forecast horizon. The 24h *lookback* windows used for features are unrelated and stay.

- Binary classification, one probability per horizon (3 outputs; one model per horizon is fine).
- Origins are on a **30-minute grid**, in UTC.
- Label = 1 if any alert overlaps `(t, t+h]`. This includes an alert already active at `t`.
- Because of that, **every metric is reported on two views**: (a) all origins, (b) only origins where Kyiv is clear at `t`. View (b) is the headline result of forecasting skill; view (a) alone is inflated by persistence, mostly at 1h and 3h.

## Data splits (chronological only, never shuffle)

Let `E` be the end of the last fully covered day in the alert data (Europe/Kyiv day boundaries). Do not use "today": the newest data is often incomplete.

- **Fixed dates (Europe/Kyiv local days, fixed 2026-10-12):**
  - **Test:** 2026-09-14 to 2026-10-11 inclusive (28 days).
  - **Validation:** 2026-08-17 to 2026-09-13 inclusive (28 days).
  - **Train:** everything before 2026-08-17 (the volunteer data starts 2022-02-25; the first train origin is provisional, see phase 1).
  - In UTC (Kyiv is UTC+3 in this period): val starts 2026-08-16 21:00, test starts 2026-09-13 21:00, test ends 2026-10-11 21:00 (exclusive).
  - **Correction (2026-10-12):** an earlier version of this file said every test origin has its full 6h label because the snapshot runs to 04:59 UTC. That was wrong. The dataset writes an alert only when it closes, so alerts still open at snapshot time are missing. Kyiv's newest row (2026-10-11 19:10 UTC) is naive, which proves a successor alert is open and absent, and it cannot have started before 22:10:32 UTC. **Kyiv is complete only until 2026-10-11 22:10:32 UTC** (`data.complete_until`).
  - **Evaluation origins are therefore limited to windows ending inside the complete data:** `t + h <= complete_until`. For the test split this drops 0 origins at 1h, 3 at 3h and 9 at 6h (1344 / 1341 / 1335 remain). Validation is unaffected. Re-download the dataset after a few days and diff the end of the test window: labels in the final hours can change retroactively.
- **Purge gap (applies to rows used for fitting, per horizon, STRICT):** a fit row with origin `t` is kept only if `t + h < ` the first origin of the split being evaluated, so the last kept row's label window ends one grid step (30 min) before the evaluation start. The gap equals the model's own horizon: 1h, 3h, 6h. A single 6h gap would be leak-free for all three, but it over-purges the 1h and 3h models, so use the per-horizon gap. Strict (not `<=`) so that an alert starting exactly at the first evaluation origin can never enter a fit label.
  - This applies when fitting on train and evaluating on val, and when fitting on train+val and evaluating on test (val rows near the test boundary are purged).
  - Evaluation origins are not purged beyond the completeness limit above. Compare horizons on the same evaluation origins where possible.
- Features may look back into earlier splits. That is real history, not leakage.
- Protocol: tune and calibrate on val -> freeze threshold and hyperparameters -> retrain once on train+val -> evaluate on test **once**. Never tune on test. No look at test metrics until the pipeline is frozen.

## Metrics

- **PR-AUC (average precision)**, **Brier score**, **precision**, **recall**, per horizon.
- Also report: **Brier skill score** vs climatology, a **reliability / calibration plot**, and **log loss**. Brier and PR-AUC depend on the base rate, so they are not comparable across horizons without a reference.
- Precision and recall need a threshold. **The main operating point is recall 0.8.** Also report precision at recall **0.5, 0.8 and 0.9**, each with the recall actually achieved (ties in scores can overshoot the target). All thresholds are chosen **on validation only**, frozen, then applied to test. On validation itself these are in-sample. Persistence is a hard 0/1 classifier and has no recall targets.
- Alerts cluster in time, so use a **block bootstrap by day** for confidence intervals, never an i.i.d. bootstrap.
- Always show baselines next to the model. They differ by view:
  - **View (a), all origins:**
    1. Persistence (alert active at `t` in the real-time state, so a naive alert counts as open). It is meaningful here: it captures ongoing alerts continuing. Its old numbers were inflated by the naive-flag leak (see rule 1).
    2. Hour-of-day (and day-of-week) climatology, fit on train only.
    3. Recent alert activity (fraction of alert time / number of alerts in the last 24h).
  - **View (b), Kyiv clear at `t`:**
    1. Climatology.
    2. Recent alert activity.
    3. Persistence is reported for completeness only. In view (b) it predicts "no alert" for every row (recall 0, PR-AUC = base rate), so it is **not a baseline to beat** there.

  A model that doesn't beat the meaningful baselines isn't a result.
- "Clear at `t`" is determined from the alert state known at `t` (see leakage rule 1): an alert that started before `t` with an unknown end counts as active.

## Leakage rules (the important part)

1. **Alert end times.** At origin `t`, an alert that started before `t` and has an `end` after `t` is **still active with unknown end**. Never build a feature from an `end` timestamp > `t`. Features must be computed from the alert state as it was at `t`.
   - **The `naive` flag is a verified leak, not just noise.** The dataset's processor (`volunteer_etryvoga_processor.py`) writes `naive=True`, `finished_at = started_at + 30 min`, only when the next activation message for the region arrives **more than 3 hours after this alert's start**; a next activation within 3h silently overwrites the open alert instead. Checked on the data: 0 of 5,828 naive alerts (all regions) have a next start within 3h (minimum gap exactly 3.000h), against 45.6% of non-naive alerts; every naive alert has duration exactly 1800s, and only 43 of 110,340 non-naive alerts do. So a 30-minute duration tells you that the region has no new alert for at least 3h after the start.
   - **Consequences and rules:** (1) features and the clear-at-t state must use the **real-time view** (`Intervals.for_region_state`): a naive alert stays open, end unknown, until the region's next activation. The dataset's ends are for **labels only** (`Intervals.for_region`). (2) Never use `naive`, or durations and ends taken from the dataset table, as features: they carry the 1800s signature. Durations, open age and time since end computed from the real-time state are legal, because a naive alert there closes only when the next activation arrives and has no 1800s signature. (3) Impact: 6.6% of train origins and 30% of validation origins lie in a naive shadow (dataset says clear, real time says open); label rates there are about half the normal clear rate in validation (1h: 0.114 vs 0.239). Inside such a shadow no alert can start before start+3h; checked on 447 train origins, 0 positives.
   - The share of naive Kyiv alerts rose from 8% overall to 18% in the last 12 months and about 25% recently, so the issue is concentrated in the regime we are tested on. Labels still use the dataset's imputed ends, so they are noisy near naive alerts (a real alert may have lasted longer than 30 min). Report a sensitivity check that excludes origins whose label window touches a naive alert.
2. **Timezones.** Store and compute in UTC. Convert to Europe/Kyiv only for calendar features (hour, weekday), because DST shifts them.
3. **Fit statistics on train only.** Scalers, encoders, imputers, target encodings, calibrators, and any threshold selection are fit on the allowed past only.
4. **Rolling and lag features** use a strictly backward window ending at `t`. Add a unit test that perturbs data after `t` and asserts the features at `t` don't change.
5. **Other regions' alerts** (neighbouring oblasts) are a **hypothesis to test, not an assumed advantage.** They are leak-free only if rule 1 holds for them too, but whether they help is an empirical question. See "Hypothesis: neighbouring-region alerts" below.
6. **Text / media (phase 4)**, same rule:
   - Use the **post time**, not the scrape time. Edits and deletions made after `t` must not be visible at `t`. If the original text can't be recovered, don't use that source.
   - Do **not** use channels that publish the alerts themselves. They are the label.
   - Useful inputs are channels reporting launches and drone/missile tracks, which precede alerts.
   - Models used to process text must be frozen feature extractors (embeddings). Don't ask a pretrained LLM to judge "was there an attack": its training data may extend past the forecast time and it may know the outcome.
   - Text exists only from some start date. Handle earlier rows with a missing indicator or start training at that date.
7. **No leakage through the pipeline.** Cross-validation inside train must be forward-chaining (expanding window) with the same purge gap. No random K-fold.

## Hypothesis: neighbouring-region alerts

Claim to test: knowing which other regions are under alert at `t` improves forecasts for Kyiv city.

- **Why it might fail:** alerts often start in several regions at nearly the same moment, so the lead time over Kyiv may be near zero. For ballistic threats it may be negative (Kyiv alerts first). It could also help only at the 1h horizon and do nothing at 3h/6h.
- **Step 1, lead-time analysis on train only (done in phase 1, before any model):** for each Kyiv alert start, find the earliest alert start in each other region within the preceding few hours. Look at the distribution of lead times per region. If it is concentrated near zero, the hypothesis is weak before any model is trained.
- **Step 2, ablation on the three evaluation windows (see Phase 3 protocol, Stage C):** the same model with and without the neighbour feature group, compared per horizon and per view. The decision rule is written down before looking at results: keep the group only if it improves PR-AUC and log loss in view (b) beyond the multi-window block-bootstrap noise.
- **Test set is not used** to decide whether the group stays. Report the final with/without comparison on test once, as a result.
- Guard against a false signal from timestamps: if the dataset's timestamps for different regions come from different sources or were reconstructed later, check the lead-time analysis for suspicious regularities (for example, a constant offset).
- Each extra hypothesis tried on val reduces how much we can trust a val win. Keep a short log of ablations run.

## Phase 1 results (scripts in `scripts/phase1/`, run with `.venv/bin/python -I`)

Base rates, 30-min origins. Test labels were not computed. **Corrected 2026-10-12:** the first version of this table defined "clear at t" from the dataset's ends and so included the naive shadow (see leakage rule 1). Clear is now the real-time state.

| Period | Clear at t (real-time) | a: 1h / 3h / 6h (all origins) | b: 1h / 3h / 6h (Kyiv clear at t) |
|---|---|---|---|
| Train overall | 88.0% | 0.112 / 0.212 / 0.330 | 0.062 / 0.166 / 0.290 |
| Last 12 months of train | 75.8% | 0.111 / 0.234 / 0.378 | 0.075 / 0.206 / 0.356 |
| Validation | 59.4% | 0.281 / 0.530 / 0.746 | 0.239 / 0.542 / 0.767 |

(The old dataset-ends view (b) gave 94.6% / 95.8% / 89.5% clear and view-(b) rates 0.061 / 0.167 / 0.292, 0.072 / 0.200 / 0.351 and 0.196 / 0.474 / 0.716.)

- **Val is a different regime from train** (the August-September surge). Base rates are 2-3x higher in val, and the share of origins that are truly clear in real time falls from 88% in train to 59% in val (more alerts, and more with no end message). Expect a calibration shift. At 6h the view-(b) base rate is 77%, so PR-AUC has a high floor there. Always judge against base rate and climatology.
- **Lead-time analysis (train only, 633 Kyiv alert starts in the last 12 months):**
  - No region shows an excess lead of 15+ minutes over a shifted-onset null (best +3 percentage points, consistent with noise across 25 regions). Kyivska oblast starts at essentially the same moment as Kyiv city (median lead about -1.5 min). The hypothesis in its strict form, "neighbours alert well before Kyiv", is weak.
  - Part A's nearest-start measure is biased towards zero lead for regions that start simultaneously; treat it as a rough indicator only.
  - **State-based signal exists for a few regions, but smaller than first reported.** The first run used the dataset's ends for "Kyiv clear" and "region active", which is the leaky state. Redone with the real-time state (2026-10-12, last 12 months of train, Kyiv clear, R active for 15+ min): P(Kyiv alert in the next 1h) is about 1.9-2.3x the base rate for Volynska, Rivnenska, Cherkaska and Zhytomyrska oblasts (earlier: up to 3.1x, with Vinnytska at the top, which no longer ranks). Kyivska oblast now shows lift below 1 (0.83). The number of active regions gives no lift until 11+ regions are active (n=98 origins). Regions that stay open from stale naive alerts dilute "active", so open-age features will matter.
  - **Caveats:** origins are autocorrelated, so the effective sample is far smaller than the origin counts; regions were picked from 25 candidates on train; the effect may differ in the current regime.
  - **Decision:** build a neighbour feature group (state and time since start per region, with strong regularization), then let the validation ablation decide. Do not trust the lifts above as final.

## Phase 2: pipeline and baselines (package `src/alertforecast/`, tests in `tests/`)

- **Layout:** `config` (paths, fixed split dates), `data` (`Intervals`: point-in-time alert state), `labels`, `splits` (grid, purge, test lock), `frame` (origin frame: features + labels), `baselines`, `metrics`, `evaluate`. Runner: `scripts/phase2/run_baselines.py` (validation only). Install once with `.venv/bin/pip install -e ".[dev]"`; run tests with `.venv/bin/python -I -m pytest -q`.
- **Conventions fixed in code and tests:** an alert is `[start, end)`; active at `t` means `start <= t < end`; the label window is `(t, t+h]` and an alert starting exactly at `t+h` counts; features are clipped at `t` (`Intervals.coverage_until`), so an alert's end after `t` is never used; the purge keeps a fit row only if `t + h < first evaluation origin` (strict), per horizon.
- **Two interval tables on purpose:** `Intervals.for_region_state` (real-time: naive alerts stay open until the next activation) feeds features and the clear-at-t state via `build_frame(t, state_iv, label_iv)`; `Intervals.for_region` (dataset ends) feeds labels only. `build_frame` adds labels only if `label_iv` is passed.
- **Test lock:** `splits.eval_origins("test")` raises unless `allow_test=True`. Only the single final evaluation may pass it.
- **Leakage tests** perturb the future (stretch the alert running at the cutoff, delete and invent later alerts) and assert that features at `t` and fit labels do not change. They were validated by planting a deliberately leaky feature, which the test catches. Keep that property when adding features: the cutoff must fall inside a running alert, otherwise end-time leaks go unnoticed.
- **Model comparison:** use the paired day-block bootstrap (`metrics.block_bootstrap_diff_ci`), not overlapping unpaired CIs. With a 28-day validation window, unpaired CIs are too wide to separate models.
- **Per-view fitting:** each view is its own task. View (b) baselines are fit on clear-at-t rows only, and Brier skill uses a climatology fitted for that view.
- **Diagnostic row:** `DIAGNOSTIC_const_val_rate` predicts validation's own base rate. It is not deployable and not a baseline to beat. It shows how much Brier is lost to the regime shift.

Validation baseline results (28 days, corrected for the naive-flag leak and the strict purge; reports in `reports/`; the earlier numbers are superseded):
- **Calibration shift dominates Brier.** In view (b) the constant val-rate diagnostic beats every baseline on Brier: 1h 0.182 vs best baseline 0.195 (recent activity), 3h 0.248 vs 0.306, 6h 0.179 vs 0.272. Models trained on the lower-rate train regime are miscalibrated on val, so Brier mostly measures that shift. Recalibration, recency weighting or a rate-tracking feature should be tested in phase 3.
- **Ranking skill in view (b) is weak for baselines.** PR-AUC, base rate in brackets: 1h climatology 0.239 [0.239], recent activity 0.275; 3h 0.536 / 0.627 [0.542]; 6h 0.782 / 0.855 [0.767]. Paired differences of recent activity vs hour-of-day climatology in view (b) are positive but their CIs include zero at every horizon (1h +0.036 [-0.025, 0.086], 3h +0.092 [-0.029, 0.174], 6h +0.074 [-0.014, 0.135]). Day-of-week climatology is clearly better only at 6h (+0.050 [0.004, 0.093]) in view (b), and at every horizon in view (a).
- **Recent activity beats climatology on Brier** in every horizon and view (paired CIs exclude zero), mostly through calibration towards the recent rate and through the open-alert signal in the real-time state.
- **Precision at recall 0.8 (view b):** recent activity 0.278 / 0.624 / 0.863 at 1h / 3h / 6h, against base rates 0.239 / 0.542 / 0.767. With recall 0.5 and 0.9 the picture is the same: baselines add only a few points of precision over the base rate.
- **Persistence is no better than the base rate once the leak is removed.** View (a) PR-AUC 0.312 / 0.523 / 0.733 vs base 0.281 / 0.530 / 0.746 (it was 0.55 at 1h with the leaky state). As a hard classifier at 1h: precision 0.343, recall 0.495. In view (b) it is degenerate by construction.

## Phase 3 protocol (approved by the user 2026-10-12, with two changes: multiple windows, and model-side adaptation before calibration)

**Problem solved:** a calibrator fit on validation predictions of the train-only model does not match the final model retrained on train+val (different score distribution, and val is in-sample for it). So the calibrator is tied to a *recipe* (features, hyperparameters, recency weighting, calibrator type), not to one fitted model.

- **Evaluation windows for choosing methods (never only validation):** three 28-day Kyiv-local windows: `val` 2026-08-17..09-13, `w1` 2026-07-20..08-16, `w2` 2026-06-22..07-19. For every window, models are fit strictly before it (strict per-horizon purge), exactly as for the real test. Report each window separately and the mean. The windows have different regimes (w1 and w2 are quieter than val), which is the point: a method must not only win on one regime. The test split stays locked.
- **Stage A, model-side adaptation, before calibration.** Fixed gradient boosting hyperparameters (no tuning in this round, to keep the number of comparisons small): `HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0)`. Grid, pre-registered: recency weighting `weight = 0.5 ** (age_days / half_life)` with age measured from the first evaluation origin, half-life in {none, 180, 60, 30} days, crossed with the feature set {base, base + recent-rate group}. Recent-rate group = alert starts and alert-time fraction over the last 7 and 28 days.
  - **Features use the real-time state only.** Durations and ends from `Intervals.for_region_state` are legal (a naive alert there has no 1800s signature, it closes at the next activation). Dataset ends (`for_region`) are forbidden as features.
  - **Decision rule (written before running):** the primary metric is **log loss in view (b), averaged over the three windows**; the secondary metric is PR-AUC in view (b). Differences vs the base configuration (no weighting, base features) use the multi-window paired day-block bootstrap (resample days within each window, average the per-window differences). A configuration is adopted only if, at **at least 2 of the 3 horizons**, its primary difference has a CI entirely below 0, **and** it is not significantly worse on the primary metric at any horizon, **and** its PR-AUC difference CI is not entirely below 0 at any horizon. If several qualify, take the one with the lowest mean primary metric; ties go to the simpler one. If none qualify, keep the base configuration. Seven comparisons are made, so treat a marginal win with suspicion; record the ablation log.
- **Stage B, calibration, on top of the Stage A winner.** For each evaluated window and horizon, build K = 3 out-of-time calibration blocks of 28 days directly before the window. Each block is predicted by the same recipe fit strictly before that block (strict purge). The pooled out-of-time predictions fit the calibrator. The window's own model is fit on all data before the window and the calibrator is applied to its predictions. Candidates, pre-registered: none, intercept shift (logit slope fixed to 1), Platt (logit slope and intercept), Platt with recency weights over blocks (1, 0.5, 0.25). Isotonic is excluded on purpose (about 4,000 clustered rows). Calibrators are fit on all rows, and calibration is reported in both views.
  - **Decision rule:** the primary metric is Brier in view (b) averaged over the windows; the secondary is log loss. Same adoption rule as Stage A (CI below 0 at at least 2 of 3 horizons, never significantly worse). PR-AUC cannot change under a monotone calibrator; this is checked.
  - **Thresholds for recall 0.5 / 0.8 / 0.9** are chosen on the pooled calibrated out-of-time predictions of the calibration blocks (about 12 weeks, less noisy than one window) and applied to the window. The recall actually achieved on the window shows how well thresholds transfer under the regime shift.
- **Stage C, neighbour features:** the earlier ablation (hypothesis section) is run on the same three windows after Stages A and B. Definition fixed before running: for every other region with at least 100 alerts in the file, two real-time-state features, `nb_open_<region>` (0/1) and `nb_age_<region>` (hours since the start of the open alert, 0 if clear), plus `nb_n_open` (number of other regions open). The recipe is the Stage A/B winner plus this group, same fixed GBM parameters, no extra regularisation tuning. Adoption rule: the Stage A rule (primary = log loss in view (b), CI below 0 at at least 2 of 3 horizons, never significantly worse on log loss, PR-AUC never significantly worse).

**Stage A result (2026-10-12): no configuration qualifies; the base recipe is kept** (no recency weighting, base features). Recency weighting is significantly worse on log loss at every horizon (mean over windows, view b, paired CI above 0), mostly on the surge window `val`: the base model already under-predicts val (mean prediction / observed rate 0.63-0.67), and weighting recent data pulls it further towards the previous quieter regime (0.38-0.55 on val). On `w1`, where the preceding weeks were similar, weighting does move the level towards 1.0 (0.82 to 0.99 at half-life 30d). The recent-rate group is neutral: log-loss difference vs base -0.003 [-0.015, 0.012] at 1h, +0.001 [-0.031, 0.032] at 3h, -0.016 [-0.070, 0.034] at 6h; PR-AUC differences also within noise. Fixed GBM hyperparameters, 8 configurations, 7 comparisons.

**Stage B result (2026-10-12): no calibrator qualifies; no calibration is applied.** Brier in view (b), difference vs none (mean over windows): intercept shift +0.0007 [0.0003, 0.0012] at 1h (significantly worse), -0.0047 [-0.0066, -0.0029] at 3h and -0.0136 [-0.0193, -0.0076] at 6h (significantly better); Platt and Platt with recency weights are within noise at every horizon. The pre-registered rule requires "never significantly worse", so intercept shift is rejected. It is a near miss to be re-checked on a fresh window, not adopted post hoc. A calibrator fit on blocks before a window cannot anticipate a regime jump: on `val`, Platt is worse than no calibration at 3h (Brier 0.2938 vs 0.2774).

**Operating points from pooled out-of-time thresholds (view b, mean over windows):** precision at recall 0.5 is 0.169 / 0.395 / 0.614 at 1h / 3h / 6h against base rates 0.150 / 0.368 / 0.559. Thresholds do not transfer well across regimes: at target recall 0.8 the achieved recall is 0.96-0.99 on `val` and `w2` but 0.76-0.81 on `w1`, with precision barely above the base rate. **The GBM adds little ranking skill over the simple baselines** (view b PR-AUC at 1h: val 0.269 vs recent activity 0.275 vs climatology 0.239; w1 0.116 vs 0.119 vs 0.135; w2 0.129 vs 0.111 vs 0.126), and its log loss is worse than a constant at the window's own rate in every window. From alert history alone, forecasting a new alert from a clear state is a weak-signal task.

**Stage C result (2026-10-12): the neighbour group is not adopted.** Log loss in view (b), difference vs base (mean over windows): +0.0044 [-0.0093, 0.0198] at 1h, +0.0144 [-0.0100, 0.0396] at 3h, +0.0289 [-0.0049, 0.0660] at 6h; no horizon is won. A hint at 1h only: PR-AUC rises in all three windows (+0.073 val, +0.002 w1, +0.026 w2; mean +0.034 [-0.008, 0.081]), but that is inside the noise, and the log loss does not improve (the 47-feature model under-predicts val even more, mean prediction / observed rate 0.50 vs 0.63). At 3h and 6h there is no sign of a benefit. Both the lead-time analysis and this ablation say the same: neighbour state adds little beyond Kyiv's own history.

**Recipe frozen after Stages A-C:** base features (hour, day of week, open state and age, time since last start and end, start counts over 1h/3h/6h/24h, alert-time fractions over 6h/24h), no recency weighting, no recent-rate group, no neighbour group, no calibration, fixed GBM hyperparameters. The pre-registered GBM-vs-logistic comparison selected the GBM (see Open decisions).

Ablation log (every comparison made on the three windows): Stage A 7 (config vs base) x 3 horizons; Stage B 3 (calibrator vs none) x 3 horizons; Stage C 1 (neighbours vs base) x 3 horizons. The Stage A/B/C winners were chosen with the pre-registered rules; none of the rules was changed after seeing results.
- **Test:** the frozen recipe (Stage A/B winners) is rolled forward by one block: the model is fit on all data before the test (train + val, strict purge), the calibrator comes from the three blocks before the test (the last of which is the validation window), thresholds likewise. The test is evaluated once.
- **Limitations to report:** a static calibrator cannot follow a regime that keeps shifting inside the test window, so report calibration drift by week. Hyperparameters are not tuned in this round.

## Test result (2026-10-12, single run, `scripts/phase5/run_test.py test`)

The frozen recipe was evaluated once on 2026-09-14..10-11; the marker `reports/.test_run_done` records the commit (`15b25a3`), that the working tree had uncommitted code changes, and a SHA-256 of all code. A dry run on validation reproduced the Stage A predictions exactly before the real run. **Nothing was changed after seeing the test results.** Details, tables and figures are in `README.md` and `reports/test_*.csv`.

- **Null result in the headline view (Kyiv clear at t):** PR-AUC 0.232 / 0.551 / 0.784 at 1h / 3h / 6h against base rates 0.227 / 0.553 / 0.791. Precision at the frozen thresholds is within 0.007 of the base rate at recall 0.8 and 0.9 and below it at recall 0.5. Frozen thresholds did not transfer (achieved recall 0.89-0.99 at targets 0.8 and 0.9).
- **Better than climatology only in Brier and log loss** (significant at all horizons); not significantly better in PR-AUC. Significantly worse than a constant at the test window's own base rate (Brier +0.013 / +0.071 / +0.105): the model under-predicts every week (mean prediction 53-88% of the observed rate).
- **No significant difference between the GBM and the recent-activity logistic baseline** in view (b) at any horizon. In view (a) the GBM's PR-AUC advantage over that baseline comes largely from the open-alert state.
- **Sensitivity to naive alerts:** excluding origins whose label window touches one leaves the conclusion unchanged.
- **Test composition:** 128 Kyiv alerts in 28 days (4.6 per day), 28% of them naive. Test origins kept: 1344 / 1341 / 1335.

## Phases

1. **Data:**
   1. Choose the source (see Open decisions): compare candidates, audit the chosen one (coverage, gaps, duplicates, end-time handling, timezone).
   2. Measure per-horizon base rates, in both views.
   3. Fix the exact split dates (train/val/test) before any analysis that uses data.
   4. **Lead-time analysis for neighbouring regions** (step 1 of the hypothesis section), on the train period only. It runs here, right after the audit and base rates, to answer how much earlier neighbouring regions alert before Kyiv. The result decides whether the neighbour features are worth building in phase 3. It is a go/no-go for building them, not proof that they help: the validation ablation still makes the final call.
2. **Pipeline + baselines:** label construction, split and purge logic, leakage tests, baselines, metric code.
3. **Alerts-only model:** gradient boosting is the first choice here (tabular, small, calibrates well). Features: time-of-day/weekday, time since last alert start/end, alert counts/durations in the last 1/3/6/24h/7d. Neighbouring-region state is added only as a tested feature group (see the hypothesis section).
4. **Text features: skipped by the user's decision (2026-10-12).** The task asked for text/media as an advantage under the same no-leakage rule; this version of the project does not use it. Report this as a limitation.
5. **Report and single test run:** the frozen recipe (GBM, base features, no recency weighting, no calibration, thresholds from the three out-of-time blocks before the test) is run on the test split exactly once (`scripts/phase5/run_test.py`, guarded by a marker file). Outputs: the metric table, calibration plots, the two-view breakdown, weekly calibration drift, the naive-alert sensitivity check, baselines, and honest limitations, in `README.md`.

## Open decisions

- **Alert data source: decided. The volunteer file is the primary source, only because it is the only one covering val and test** (not because it is known to be more accurate). Candidates were compared as required above; findings (audited 2026-10-12, raw files in `data/raw/vadimkin_2026-10-12/`):
  - **Vadimkin/ukrainian-air-raid-sirens-dataset, volunteer file** (`volunteer_data_en.csv`): runs to 2026-10-12, so it is the only repo source covering the val and test windows. Region `Kyiv City` is separate from `Kyivska oblast`. Times are UTC. Source: eTryvoga channel, oblast level only.
  - **Same repo, official file** (`official_data_en.csv`): **stale, ends in early September 2026** (last row 2026-09-07, last Kyiv City alert 2026-09-05; file last changed 2026-09-08; upstream issues #6 and #7 open). Cannot cover val/test. Kyiv city appears at oblast level only. About 45% of its Kyiv rows are exact duplicates. **It has a feed failure on 2026-08-30 to 2026-09-01:** zero Kyiv alerts although alerts did occur (confirmed by the user against the files). Do not use it for training or evaluation.
  - **alerts.in.ua API:** needs a token (form at alerts.in.ua/api-request, limits 8-12 req/min), Kyiv City is UID 31, and the **history endpoint only goes back one month (`month_ago`)**. Usable as a cross-check on roughly the last 30 days and for collecting data going forward, not for the full history.
  - **Sources disagree a lot** (28 days, 2026-08-09 to 2026-09-06): only 32% of alert-minutes overlap; about 20% of 30-min origins get a different label depending on the source. Part of this is the official feed failure above, so the overlap number overstates the volunteer file's own noise. Neither file is verified ground truth; the volunteer file's own errors (for example, missed days) are not ruled out. Metrics must be reported as "relative to the volunteer file's labels".
  - **Regime shift is real, not a data error:** Kyiv alerts per month rose sharply in August-September 2026 compared with Feb-Jul 2026 (more alerts actually happened). Val and test sit in a busier regime than most of train. Treat it as genuine non-stationarity: report per-split base rates, check calibration drift, and consider recency weighting as a tested option, never as a data fix.
  - **Candidates are compared on the same recent days** (for example, the most recent 28 days of complete data in all candidates), not on different periods.
  - **Disagreements are reported, not silently resolved:** missing or extra alerts, start/end time differences (with their distribution), and merged or split alerts. The chosen source and the reason for choosing it are recorded.
- **Operating threshold:** the main operating point is recall 0.8, with precision also reported at recall 0.5 and 0.9, all chosen on validation (decided by the user). Revisit only if the use case changes.
- **Final model choice before the test: rule fixed by the user (2026-10-12), before running.** The final model is the **recent-activity logistic baseline** (`baselines.RecentActivity`: logistic regression on `frac_alert_24h` and `log1p(n_starts_24h)`) **unless the GBM's log-loss CI is entirely below 0 at 2 of the 3 horizons.** Implementation, fixed before running: three windows (`val`, `w1`, `w2`); metric = log loss in view (b); difference = GBM minus logistic, with the multi-window paired day-block bootstrap (300 resamples, seed 0, 95% CI); the GBM is the frozen Stage A-C recipe fit on all purged rows; the logistic is fit on the clear-at-t purged rows only (its per-view definition), and both are scored on the clear rows of each window. The view (b) result decides for both views. If the logistic wins, view (a) uses the same logistic fit on all rows.
  - **Result (2026-10-12): the GBM is the final model.** Log loss in view (b), GBM minus logistic, mean over windows: 1h -0.0058 [-0.0141, 0.0008] (borderline, not entirely below 0), 3h -0.0211 [-0.0383, -0.0066], 6h -0.0265 [-0.0477, -0.0066]. The CI is entirely below 0 at 2 of 3 horizons, so the rule picks the GBM. The gain is in log loss and Brier; PR-AUC differences are within noise (+0.003, +0.014, -0.006), so the GBM is better mostly through calibration and level, not ranking.
- **Intercept shift near miss:** significantly better Brier at 3h and 6h, significantly worse at 1h. The rule rejected it. If the user wants a horizon-specific rule, confirm it on a fresh earlier window (for example 2026-05-25..06-21), never on the three windows already used.
- **Snapshot refresh:** re-download the dataset after a few days and diff the end of the validation and test windows. Alerts missing from the file at snapshot time (open alerts) can appear later, so recent labels may change.
- **Cross-check against alerts.in.ua: deferred by the user ("not now").** Until it is done, the report must say that test labels come from a single unverified source. It can be added later without changing the pipeline.
- **Telegram channels and API access** for phase 4 (credentials, channel list, historical backfill and whether original post text survives).
- Whether to also retrain periodically within the test window. The default is a fixed model for all 28 days. This is slightly pessimistic compared with production and must be mentioned in the report.

## Working agreements for Claude

- The user wants **explanations for choices**. When making a modeling or data decision, say why in a sentence or two.
- **Flag anything that looks wrong or unrealistic** instead of silently going along with it.
- **Ask before** downloading large data, or calling anything that needs credentials or posts to a third party.
- **Secrets:** API tokens and credentials (alerts.in.ua token, Telegram `api_id`/`api_hash`/session files) live only in `.env`, which must be listed in `.gitignore` (also ignore Telegram session files). Load them from the environment, never hardcode them, never print them in logs or output, and never write them into this file or any tracked file. Keep a `.env.example` with variable names and empty values.
- Phase 3 modeling code waits for the user's approval of the calibration protocol. Keep dependencies small (pandas, numpy, scikit-learn, LightGBM, matplotlib).
- Put raw data in `data/raw/` (never edit it), derived data in `data/processed/`, and keep both out of git.
- Python environment is `.venv/` in the project root (gitignored). Run scripts with `.venv/bin/python -I`. Add a dependency only when needed and tell the user.
- Fix random seeds. Report the exact date ranges of each split with every result.
