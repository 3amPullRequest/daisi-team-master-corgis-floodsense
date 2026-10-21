# Demo Day notes

## Does the tide help? FloodSense with and without tide levels

**Question.** Gold now gives the model the predicted tide at Tanjong Pagar for every 5-minute tick (`gold.tide_levels` → `gold.area_features.tide_height_m` and `minutes_to_high_tide`). Do these two inputs help the gated LightGBM pair warn of PUB alerts earlier, or are they noise?

**Method.** We fit the same model twice on the same rows and change only the inputs: once with the tide and once with the tide group (`contract.FEATURE_GROUPS["tide"]`) dropped (`gated.Settings(drop=...)`). Both fits are judged out of fold on the same scoreboard over the development period, 27 Oct 2021 to 31 Dec 2024. We compare them three ways:

- threshold-free scores of the alert probability;
- PUB events flagged before PUB's own alert, with each fit warning on the same share of ticks;
- the same events paired one to one across the two fits.

We ran each comparison with three seeds.

> **Status: a laptop stand-in, not the final answer.** The team workspace wasn't reachable, so these numbers come from `explorations/tide_ablation_local.py` on landed files. It runs the production model and scoreboard code, but with two simplifications:
> - **One island-wide "area".** Placing alerts in NEA's 47 areas needs silver's gazetteer, so here the label is "a PUB alert anywhere in Singapore in the next hour".
> - **A plain 10-minute reading lag.** Gold also holds a reading back until NEA has published it; the stand-in doesn't.
>
> The answer to quote on Demo Day is the per-area run on gold: `databricks bundle run train -t dev -p flood_dev`, which now compares with and without the tide by default. It is still pending workspace access.

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

**Reading.** Over 2021–2024 the tide gives a small, consistent edge in ranking and a few extra storms caught early at a fixed warning budget. The edge is well inside the noise of 89 events.

### What this means for feature selection

1. **Keep `minutes_to_high_tide`.** The model uses it and it never made the results worse. Re-judge it on the per-area run in `flood-train`.
2. **Remove the "risk only rises with tide height" constraint on `tide_height_m`** (`gated.RISING`). The data show alerts more common at *low* tide, so the constraint stops the model using height at all, which is why it got 0.3% of the gain. Refit without the constraint before deciding whether height stays.
3. **Promote a with-tide model only if the per-area run agrees.** It must flag more events before PUB within the rule's false-alarm budget (the `beats_rule` check). If it doesn't, train with `compare_without=tide` and keep the simpler pair.

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
