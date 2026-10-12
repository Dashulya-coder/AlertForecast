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
  - An origin is kept as long as its full h-hour label window lies inside data that exists. The snapshot ends 2026-10-12 04:59 UTC, so every test origin has its full 6h label. This replaces the earlier idea of cutting the last 6h.
- **Purge gap (applies to rows used for fitting, per horizon):** a training row with origin `t` is dropped if `t + h` reaches the first origin of the split being evaluated. So the gap equals the model's own horizon: 1h, 3h, 6h. A single 6h gap would be leak-free for all three, but it over-purges the 1h and 3h models, so use the per-horizon gap.
  - This applies when fitting on train and evaluating on val, and when fitting on train+val and evaluating on test (val rows near the test boundary are purged).
  - Evaluation origins are not purged. Every val/test origin is kept as long as its label is observable. Compare horizons on the same evaluation origins.
- Features may look back into earlier splits. That is real history, not leakage.
- Protocol: tune and calibrate on val -> freeze threshold and hyperparameters -> retrain once on train+val -> evaluate on test **once**. Never tune on test. No look at test metrics until the pipeline is frozen.

## Metrics

- **PR-AUC (average precision)**, **Brier score**, **precision**, **recall**, per horizon.
- Also report: **Brier skill score** vs climatology, a **reliability / calibration plot**, and **log loss**. Brier and PR-AUC depend on the base rate, so they are not comparable across horizons without a reference.
- Precision and recall need a threshold. Choose it **on validation only** (for example, the threshold that reaches a target recall), freeze it, then apply it to test.
- Alerts cluster in time, so use a **block bootstrap by day** for confidence intervals, never an i.i.d. bootstrap.
- Always show baselines next to the model. They differ by view:
  - **View (a), all origins:**
    1. Persistence (alert active at `t`). It is meaningful here: it captures ongoing alerts continuing.
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
   - **Volunteer data has a `naive` flag.** `naive=True` means no end message was ever seen and `finished_at = started_at + 30 min` is an imputed value. The flag is only known after the fact, so **never use `naive` as a feature**. In recent months about 15-30% of Kyiv alerts are naive, so labels near them are noisy. Report a sensitivity check that excludes origins whose label window touches a naive alert.
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
- **Step 2, ablation on validation:** the same model with and without the neighbour feature group, compared per horizon and per view. The decision rule is written down before looking at results: keep the group only if it improves validation PR-AUC and Brier in view (b) beyond the block-bootstrap noise.
- **Test set is not used** to decide whether the group stays. Report the final with/without comparison on test once, as a result.
- Guard against a false signal from timestamps: if the dataset's timestamps for different regions come from different sources or were reconstructed later, check the lead-time analysis for suspicious regularities (for example, a constant offset).
- Each extra hypothesis tried on val reduces how much we can trust a val win. Keep a short log of ablations run.

## Phase 1 results (scripts in `scripts/phase1/`, run with `.venv/bin/python -I`)

Base rates, 30-min origins. Test labels were not computed.

| Period | Clear at t | a: 1h / 3h / 6h (all origins) | b: 1h / 3h / 6h (Kyiv clear at t) |
|---|---|---|---|
| Train overall | 94.6% | 0.112 / 0.212 / 0.330 | 0.061 / 0.167 / 0.292 |
| Last 12 months of train | 95.8% | 0.111 / 0.234 / 0.378 | 0.072 / 0.200 / 0.351 |
| Validation | 89.5% | 0.281 / 0.529 / 0.746 | 0.196 / 0.474 / 0.716 |

- **Val is a different regime from train** (the August-September surge). Base rates are 2-3x higher in val. Expect a calibration shift. At 6h the view-(b) base rate is 72%, so PR-AUC has a high floor there. Always judge against base rate and climatology.
- **Lead-time analysis (train only, 633 Kyiv alert starts in the last 12 months):**
  - No region shows an excess lead of 15+ minutes over a shifted-onset null (best +3 percentage points, consistent with noise across 25 regions). Kyivska oblast starts at essentially the same moment as Kyiv city (median lead about -1.5 min). The hypothesis in its strict form, "neighbours alert well before Kyiv", is weak.
  - Part A's nearest-start measure is biased towards zero lead for regions that start simultaneously; treat it as a rough indicator only.
  - **State-based signal exists for a few regions.** With Kyiv clear at t and region R active for 15+ min, P(Kyiv alert in the next 1h) is 2-3x the base rate for Cherkaska, Rivnenska, Zhytomyrska and Volynska oblasts. Kyivska oblast shows no lift (about 1.0). The number of active regions alone gives no lift until 6+ regions are active, so this is not just a generic "wave" indicator.
  - **Caveats:** origins are autocorrelated, so the effective sample is far smaller than the origin counts; regions were picked from 25 candidates on train; the effect may differ in the current regime.
  - **Decision:** build a neighbour feature group (state and time since start per region, with strong regularization), then let the validation ablation decide. Do not trust the lifts above as final.

## Phases

1. **Data:**
   1. Choose the source (see Open decisions): compare candidates, audit the chosen one (coverage, gaps, duplicates, end-time handling, timezone).
   2. Measure per-horizon base rates, in both views.
   3. Fix the exact split dates (train/val/test) before any analysis that uses data.
   4. **Lead-time analysis for neighbouring regions** (step 1 of the hypothesis section), on the train period only. It runs here, right after the audit and base rates, to answer how much earlier neighbouring regions alert before Kyiv. The result decides whether the neighbour features are worth building in phase 3. It is a go/no-go for building them, not proof that they help: the validation ablation still makes the final call.
2. **Pipeline + baselines:** label construction, split and purge logic, leakage tests, baselines, metric code.
3. **Alerts-only model:** gradient boosting is the first choice here (tabular, small, calibrates well). Features: time-of-day/weekday, time since last alert start/end, alert counts/durations in the last 1/3/6/24h/7d. Neighbouring-region state is added only as a tested feature group (see the hypothesis section).
4. **Text features:** add Telegram/news after phase 3 and report the gain over the alerts-only model on validation, then test.
5. **Report:** the metric table, calibration plots, the two-view breakdown, and honest limitations.

## Open decisions

- **Alert data source: decided. The volunteer file is the primary source, only because it is the only one covering val and test** (not because it is known to be more accurate). Candidates were compared as required above; findings (audited 2026-10-12, raw files in `data/raw/vadimkin_2026-10-12/`):
  - **Vadimkin/ukrainian-air-raid-sirens-dataset, volunteer file** (`volunteer_data_en.csv`): runs to 2026-10-12, so it is the only repo source covering the val and test windows. Region `Kyiv City` is separate from `Kyivska oblast`. Times are UTC. Source: eTryvoga channel, oblast level only.
  - **Same repo, official file** (`official_data_en.csv`): **stale, ends in early September 2026** (last row 2026-09-07, last Kyiv City alert 2026-09-05; file last changed 2026-09-08; upstream issues #6 and #7 open). Cannot cover val/test. Kyiv city appears at oblast level only. About 45% of its Kyiv rows are exact duplicates. **It has a feed failure on 2026-08-30 to 2026-09-01:** zero Kyiv alerts although alerts did occur (confirmed by the user against the files). Do not use it for training or evaluation.
  - **alerts.in.ua API:** needs a token (form at alerts.in.ua/api-request, limits 8-12 req/min), Kyiv City is UID 31, and the **history endpoint only goes back one month (`month_ago`)**. Usable as a cross-check on roughly the last 30 days and for collecting data going forward, not for the full history.
  - **Sources disagree a lot** (28 days, 2026-08-09 to 2026-09-06): only 32% of alert-minutes overlap; about 20% of 30-min origins get a different label depending on the source. Part of this is the official feed failure above, so the overlap number overstates the volunteer file's own noise. Neither file is verified ground truth; the volunteer file's own errors (for example, missed days) are not ruled out. Metrics must be reported as "relative to the volunteer file's labels".
  - **Regime shift is real, not a data error:** Kyiv alerts per month rose sharply in August-September 2026 compared with Feb-Jul 2026 (more alerts actually happened). Val and test sit in a busier regime than most of train. Treat it as genuine non-stationarity: report per-split base rates, check calibration drift, and consider recency weighting as a tested option, never as a data fix.
  - **Candidates are compared on the same recent days** (for example, the most recent 28 days of complete data in all candidates), not on different periods.
  - **Disagreements are reported, not silently resolved:** missing or extra alerts, start/end time differences (with their distribution), and merged or split alerts. The chosen source and the reason for choosing it are recorded.
- **Cross-check against alerts.in.ua: deferred by the user ("not now").** Until it is done, the report must say that test labels come from a single unverified source. It can be added later without changing the pipeline.
- **Telegram channels and API access** for phase 4 (credentials, channel list, historical backfill and whether original post text survives).
- Whether to also retrain periodically within the test window. The default is a fixed model for all 28 days. This is slightly pessimistic compared with production and must be mentioned in the report.

## Working agreements for Claude

- The user wants **explanations for choices**. When making a modeling or data decision, say why in a sentence or two.
- **Flag anything that looks wrong or unrealistic** instead of silently going along with it.
- **Ask before** downloading large data, or calling anything that needs credentials or posts to a third party.
- **Secrets:** API tokens and credentials (alerts.in.ua token, Telegram `api_id`/`api_hash`/session files) live only in `.env`, which must be listed in `.gitignore` (also ignore Telegram session files). Load them from the environment, never hardcode them, never print them in logs or output, and never write them into this file or any tracked file. Keep a `.env.example` with variable names and empty values.
- Don't write pipeline code until phase 1 is agreed. Keep dependencies small (pandas, numpy, scikit-learn, LightGBM, matplotlib).
- Put raw data in `data/raw/` (never edit it), derived data in `data/processed/`, and keep both out of git.
- Python environment is `.venv/` in the project root (gitignored). Run scripts with `.venv/bin/python -I`. Add a dependency only when needed and tell the user.
- Fix random seeds. Report the exact date ranges of each split with every result.
