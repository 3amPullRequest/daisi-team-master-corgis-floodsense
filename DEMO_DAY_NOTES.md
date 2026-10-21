# Demo Day notes

## Judging criteria

| Criterion | Weight | What judges look for |
|---|---|---|
| Social impact | 30% | Who benefits, how much, and whether the solution could realistically be adopted. |
| Technical execution on Databricks | 30% | A working end-to-end pipeline: ingestion, transformation, governance, analytics or AI. |
| Solution & user experience | 20% | A usable dashboard, app or interface a non-technical stakeholder could act on. |
| Data rigour | 10% | Sound use of Singapore open data, handling of messy data, and honest limitations. |
| Presentation | 10% | A clear story, a confident live demo and a well-handled Q&A. |

## Does the tide help? FloodSense with and without tide levels

**Question.** Gold now gives the model the predicted tide at Tanjong Pagar for every 5-minute tick (`gold.tide_levels` → `gold.area_features.tide_height_m` and `minutes_to_high_tide`). Do these two inputs help the gated LightGBM pair warn of PUB alerts earlier, or are they noise?

**Method.** We fit the same model twice on the same rows and change only the inputs: once with the tide and once with the tide group (`contract.FEATURE_GROUPS["tide"]`) dropped (`gated.Settings(drop=...)`). Both fits are judged out of fold on the same scoreboard, and the PUB events are paired one to one across the two fits.

### The answer: per-area, on gold (`flood-train`, Databricks, 21 Oct 2026)

`flood-train` refits the chosen pair without the tide in a nested MLflow run (`compare_without=tide`). It used all 47 areas, 12.3 million training area-ticks (`gold.model_rows`, split `train`) and the development period 27 Oct 2021 to 19 Dec 2025, with 5 blocked day-folds. Both fits use gate recall 0.9 and the rule's false-alarm budget (0.84 per area-month).

| Out of fold, 227 judged PUB events on 105 days | With tide | Without tide | Difference |
|---|---|---|---|
| PUB events flagged before PUB's own alert | **20.7%** (95% CI 15.9–26.4%) | 17.2% (12.8–22.6%) | **+3.5 pts** |
| Mean lead before PUB's alert (misses count 0) | 4.9 min | 4.6 min | +0.3 min (95% CI −0.9 to +1.5) |
| False alarms per area-month | 0.79 | 0.72 | +0.06 (both within the 0.84 budget) |
| Hours under a warning per area-month | 1.19 | 1.08 | +0.11 |
| Area-ticks the gate passes | 12.5% | 11.0% | +1.6 pts |

- **Paired events:** 35 events were flagged early by both fits, 12 only with the tide and 4 only without it. That leans towards the tide but falls just short of the usual significance bar (exact McNemar p = 0.077).
- **What the model leans on:** the tide took 10% of the rater's gain and 7% of the gate's.
- **Against the bar:** the with-tide pair beats the rainfall rule (20.7% flagged early against 9.7%, 4.9 min lead against 0.9, at 0.79 false alarms against 0.84). It is registered as version 1, `@challenger` and `@champion`, and `flood-live` now scores with it.
- **Flash floods:** only 1 flood event could be judged in the development period (30 fell in missing label windows), so this comparison says nothing about High.

**Reading.** Per area, the tide helps the model catch more storms early (+3.5 points, 16 discordant events splitting 12 to 4) for slightly more warning time. The evidence is suggestive, not conclusive: p = 0.077, and the lead interval includes 0. It agrees with the laptop stand-in below, which used one island-wide area and found the same direction.

### Supporting evidence: the laptop stand-in (before workspace access)

These numbers come from `explorations/tide_ablation_local.py` on landed files, run before the workspace was reachable. It runs the production model and scoreboard code, but with two simplifications, so treat it as a cross-check of the per-area result above:

- **One island-wide "area".** Placing alerts in NEA's 47 areas needs silver's gazetteer, so here the label is "a PUB alert anywhere in Singapore in the next hour".
- **A plain 10-minute reading lag.** Gold also holds a reading back until NEA has published it; the stand-in doesn't.

Its development period is 27 Oct 2021 to 31 Dec 2024, and each comparison ran with three seeds, compared three ways: threshold-free scores, events flagged at a fixed share of warned ticks, and paired events.

### Data (stand-in)

| | |
|---|---|
| Ticks with both labels and features recorded | 220,074 |
| Ticks followed by an alert within the hour | 1,460 (0.66%) |
| PUB alert locations, Telegram 2021–2024 (incl. 872 posts recovered from the web archive) | 561 |
| PUB events judged (flash floods among them) | 89 (5) |
| Days left out: labels missing / features missing | 318 / 109 |

### Results (stand-in, mean of 3 seeds)

| | With tide | Without tide | Difference |
|---|---|---|---|
| ROC-AUC (out of fold) | 0.9279 | 0.9275 | +0.0004 (higher with tide in 3/3 seeds) |
| Log loss | 0.02462 | 0.02471 | −0.00009 (lower with tide in 3/3 seeds) |
| Average precision | 0.188 | 0.184 | +0.004 (2 of 3 seeds better) |
| **Warning on 2% of ticks:** events flagged before PUB | 83.1% | 80.5% | +2.6 pts |
| — mean lead before PUB's alert | 38.1 min | 34.2 min | +3.9 min |
| **Warning on 5% of ticks:** events flagged before PUB | 98.9% | 95.5% | +3.4 pts |
| — mean lead before PUB's alert | 80.9 min | 80.9 min | 0 |
| — false alarms per month (island-wide) | 22.6 | 22.4 | +0.2 |

- **Paired events, 5% budget, per seed:** 2, 4 and 3 events were flagged early only with the tide, and none only without it. That isn't significant (exact McNemar p = 0.5, 0.125, 0.25). The lead difference ranged from −2.8 to +3.3 min, and every 95% day-bootstrap interval included 0.
- **What the model leans on:** the tide took 14–18% of the rater's gain, almost all of it through `minutes_to_high_tide`. `tide_height_m` got 0.3%.
- **The tide's link to alerts runs the other way from expected.** Alerts were twice as common at the lowest tide quartile as at the highest: 1.01% against 0.48% of ticks. They were also highest at the lowest quartile within the 12:00–20:00 storm hours alone: 2.3% against 1.0–1.4% for the other three. That is an association, not proof of a cause; the season or the hour may be what links them.

**Reading (stand-in).** Over 2021–2024 the tide gives a small, consistent edge in ranking and a few extra storms caught early at a fixed warning budget. The edge is well inside the noise of 89 events.

### What this means for feature selection

1. **Keep the tide group.** Per area, on gold, it adds 3.5 points of PUB events caught early within the false-alarm budget, and the stand-in agrees on direction. The champion now in production reads it.
2. **Remove the "risk only rises with tide height" constraint on `tide_height_m`** (`gated.RISING`). The stand-in found alerts more common at *low* tide (1.01% of ticks in the lowest quartile against 0.48% in the highest), so the constraint may stop the model using height at all. Refit without it and compare with `compare_without=tide` again.
3. **Re-judge on more events before quoting a firm number.** The per-area result is p = 0.077 on 227 events. The single look at the 2026 test split (`score_test=true`) adds independent events, and each later `flood-train` run logs the same comparison automatically.

### How Databricks made this comparison possible

- **Lakeflow Declarative Pipelines.** They turn the two raw tide sources into a feature without hand-run steps. The UHSLC gauge and NEA's tide table land in a Unity Catalog volume. They flow bronze → silver (`tide_gauge_hourly`, `tide_extremes`) → gold (`tide_model`, `tide_datum`, `tide_levels`) as materialized views, which rebuild when NEA's table changes. Expectations stop the pipeline from feeding the model a bad tide:
  - `tide_datum` checks the model against NEA's highs and lows (within 0.15 m and 30 min);
  - `area_features` checks that every tick has a tide.
  
  The pipeline's serverless environment installs utide, so the tide model fits inside the pipeline.
- **One gold feature table for both fits.** Both arms read the same `gold.area_features` rows and the same labels (`gold.model_rows`). The only thing that differs is which columns are dropped, so any difference is the tide's.
- **MLflow tracking.** `flood-train --compare-without tide` refits the chosen pair without the tide in a nested run under the main run. It logs:
  - `without_tide_comparison.json` (each metric with, without, the difference and which side is better);
  - paired-event metrics (`without_tide_alert_flagged_only_with`, `…_mcnemar_p`, the lead difference with its interval);
  - `without_tide_delta_*` for every headline metric;
  - the tide's gain share.
  
  The experiment UI then puts the two fits side by side, run after run, so a feature decision rests on logged evidence rather than a notebook someone remembers.
- **Lakeflow Jobs.** `compare_without` is a `flood-train` job parameter, so testing any group (`tide`, `forecast`, `rain`) is one command: `databricks bundle run train -t dev -p flood_dev --params compare_without=tide,forecast`. The job refreshes gold before training, so every comparison reads the current data.
- **Unity Catalog.** It tracks lineage from the landed files to the registered model. A pair is saved with the list of inputs it reads, so a pair without the tide loads and serves like any other. The `@challenger` / `@champion` aliases let whichever version wins the comparison be promoted without touching the app.
