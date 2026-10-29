# FloodSense model architecture

*23 October 2026. This is a review of the model pipeline: what was built, and what the data said about each
recommendation. [DEVELOPER_NOTES.md](DEVELOPER_NOTES.md) keeps the run-by-run log. The code docstrings
(`ml/ordinal.py`, `ml/gated.py`, `ml/train.py`) have the details.*

## Summary

FloodSense rates each of NEA's 47 forecast areas every 5 minutes as Low, Medium or High, by the chance of a PUB alert in
the next hour. The review proposed:

- an ordinal target;
- no gate in front of the model;
- dew point instead of relative humidity;
- 3-, 6- and 24-hour rain totals;
- a rule-based safety override;
- class weights favouring floods over near misses;
- penalized logistic regression, TabPFN, CatBoost or a Bayesian hierarchical model;
- judging on minority-class recall and PR-AUC.

Everything that runs on Free Edition was built and tested against the current model, v5 (the gated LightGBM pair).
Every model was fitted out of fold on the same five time blocks and judged by the same scoreboard, at the same
false-alarm budget (the rainfall rule's 0.87 per area-month).

**Verdict:**

1. **On the ranking the team agreed on, nothing beat v5.** The ranking counts PUB events flagged before PUB's first
   alert, a flash flood counting 10. v5's design scored 0.303. The best new design scored 0.298, a tie within noise.
   v5 stays `@champion`, and `gated` (v5's design) is `flood-train`'s default.
2. **The gate is not what held the model back.** Without it, the tree models flagged 17–22% of PUB events against v5's
   30%. The gate works as hard-negative mining: the rater learns the fine differences between storms. Its one real
   flaw is that a tick it screens out can never be warned. That is now fixable without losing the gate (the hybrid
   below).
3. **On the Round 1 flood targets, two new designs are better than v5.** Both add a flood rung fitted by penalized
   logistic regression. They flag 5 of the 13 judged development floods before PUB, as the rule does, against v5's 4.
   They also raise the public's mean warning before a flash flood from 27 minutes to 34–36, past the 30-minute target.
   The cost is 11–13 of 322 alert events. Whether that trade is worth it is the team's call. With only 13 floods,
   neither side's lead is statistically safe.
4. **Antecedent rain is not the missing signal.** Given the last hour's rain, more rain in the hours before means
   *fewer* alerts: PUB alerts as a storm starts, and Singapore's drains empty fast. As a model input it changed results
   by under a point.
5. **What didn't help:** weighting floods above near misses (−3 to −5 points of events); a flood rung fitted by
   LightGBM (179 flood rows are too few for trees); the rule as a safety override (no threshold flagged more within the
   budget); and replacing humidity with the dew point in the tree models.

The hybrid (`ordinal:alert=gated:flood=logistic`) is registered as `@challenger` (v8). It passed all seven
`flood-verify` checks, so the new serving path works end to end on Databricks:

- the pyfunc loaded from Unity Catalog matches the model exactly;
- the model's logged code runs alone, which is how the app loads it;
- no chance falls with more rain;
- both priors are finite.

`@champion` is final_2 (v14) since 21:31 SGT on 25 Oct: v12's design with the tiers as fixed chances, Low below 3%, Moderate from 3% and Elevated from 5% (decision 9). final_1 (v13), the same with Moderate from 1%, was `@champion` from about 20:36 (decision 8). The app says Moderate and Elevated; the data still says Medium and High. v12 (v10's design plus the drains' response and how far each drain's flow is above its highest earlier in the storm, `gated:add=storm+surplus+drain+rise`) was `@champion` from 19:24 that evening. v10 (`gated:add=storm+surplus`, the storm so far) was `@champion` from 24 Oct; v11, v10 with the tiers as fixed chances (Medium from 1%, High from 5%), for half an hour that night. v5 was `@champion` before v10, and the hybrid for a few hours on 23 Oct; see [Decisions](#decisions).
## Results

Out of fold, over the development period (27 Oct 2021 to 18 Dec 2025). 322 PUB events and 13 flash floods could be
judged; the others fell in missing label windows. Each model is held to the rule's false-alarm rate. "Base" features
are what v5 reads. "Extended" swaps humidity for the dew point and adds the prior rain. Specs are as logged on 23 Oct,
when `legacy` meant base and no suffix meant extended.

| model (spec) | gate | features | PUB events flagged before PUB (95% CI) | floods flagged before PUB, of 13 | public warning before a flood, min (95% CI) | false alarms per area-month | rank score |
|---|---|---|---|---|---|---|---|
| rainfall rule (`rain_15min_mm` ≥ 25) | – | – | 18.0% (14.2–22.6) | 5 | 27.6 (14.9–42.5) | 0.87 | – |
| **v5's design: gated pair** (`gated:legacy`) | yes | base | **30.1%** (25.4–35.3) | 4 | 27.1 (13.1–42.3) | 0.84 | **0.303** |
| gated pair (`gated`) | yes | extended | 25.5% (21.0–30.5) | 4 | 27.5 (13.3–43.0) | 0.82 | 0.269 |
| **hybrid: gated alert rung + logistic flood rung** (`ordinal:alert=gated:flood=logistic:legacy`) | alert rung | base | 26.7% (22.2–31.8) | **5** | **36.2** (16.7–59.7) | 0.84 | 0.298 |
| same (`ordinal:alert=gated:flood=logistic`) | alert rung | extended | 23.6% (19.3–28.5) | 5 | 36.2 (16.7–59.7) | 0.81 | 0.276 |
| hybrid with a LightGBM flood rung (`ordinal:alert=gated`) | alert rung | extended | 21.7% (17.6–26.6) | 3 | 28.1 (12.4–45.2) | 0.80 | 0.221 |
| ordinal, both rungs logistic (`ordinal:alert=logistic:flood=logistic`) | no | extended | 26.1% (21.6–31.1) | 5 | 34.3 (16.7–54.8) | 0.82 | 0.294 |
| … without the prior rain (`…:drop=antecedent`) | no | extended − prior rain | 25.5% (21.0–30.5) | 5 | 36.2 (16.7–59.7) | 0.85 | 0.289 |
| ordinal, both rungs LightGBM (`ordinal`) | no | extended | 19.9% (15.9–24.6) | 3 | 31.6 (12.9–52.4) | 0.81 | 0.207 |
| ordinal, logistic flood rung (`ordinal:flood=logistic`) | no | extended | 22.0% (17.9–26.9) | 3 | 34.2 (13.1–59.7) | 0.79 | 0.223 |
| ordinal, flood rows weigh 3 (`ordinal:flood_weight=3`) | no | extended | 17.4% (13.6–21.9) | 3 | 31.2 (11.9–53.3) | 0.83 | 0.189 |

- **Rank score** is how `flood-train` ranks fits: the share of PUB events flagged before PUB, a flood counting 10 (439
  weighted events).
- **Public warning** is the mean of whichever warned first, FloodSense or PUB. PUB alone gives 19.9 minutes before
  these floods.
- **The overrides.** Every fit in the third run tried the rule as its override at 25, 37.5, 50 and 75 mm in 15 minutes,
  and none was kept. At 25 mm the budget can't hold it. On v5's design, 37.5–75 mm costs 3 points of events (30.1% →
  27.0%) and catches no flood the pair misses.
- **Threshold-free:** row-level PR-AUC for alerts is 0.024–0.027 for every model, about 90 times the 0.03% base rate.
  The models rank rows alike; they differ in what crosses the budget. For floods, v5's own alert chance ranks flood
  rows best: 0.0054 against 0.0020–0.0029 for the flood rungs, a base rate of 0.0013%. The flood rung's value is in
  the warnings it adds, not in better ranking.
- **Looser budgets:** at 2 false alarms per area-month the designs flag 38–47% of events, and at 4, 65–73%. The
  budget remains the biggest lever on recall, as on 22 Oct.
- **The 2026 floods** (`flood-verify`, run `verify v8`). The test was spent on 19 Oct, so this is descriptive only.
  Of 5 judged floods:
  - the hybrid warned 2: Tanglin on 22 Sep, 36 minutes ahead and before PUB (v5: 26), and Woodlands on 30 Sep, which
    v5 missed. Its alert chance there is v5's, so the flood rung raised that warning;
  - v5 warned 1;
  - the rule warned all 5.

  Lim Chu Kang (30 Sep, 85 mm in the hour) is still a miss at 0.9%.
- Runs: MLflow parents `f568c845` (the first sweep; its last fit ran out of memory, so it was rerun with less
  copying), `4ee3b986`, `3f303a06`, and `2ea6e7cc` (the hybrid alone, registered as v8). The fits are deterministic:
  v5's design and the logistic model gave identical numbers in every run they were in.

## What changed, and why

### Two rungs, and a flood rung that can warn past the gate

`ml/ordinal.py` models the ordinal ladder (no alert < a warning, the near miss < a flash flood) as two binary rungs
(Frank & Hall):

- the **alert rung** gives P(a PUB alert of either kind within the hour), for Medium;
- the **flood rung** gives P(a flash flood within the hour), for High.

Each rung has its own credibility prior. The flood rung's comes from flood days, which are so few (k = 3,622 days
against about 1,074 recorded per area) that most areas get the island's rate. So an area without a water-level sensor,
where PUB can't warn, isn't taken for one where nothing floods. Each rung's learner is LightGBM, penalized logistic
regression (`ml/logistic.py`), or for the alert rung the whole gated pair (`alert=gated`, the hybrid). The flood rung
always reads every row.

**A tick is Medium when p_alert + 9 × p_flood passes the cut-off.** This is the expected value of a warning, a flood
flagged counting 10 events, as when cut-offs are chosen. So a tick the gate screened out (p_alert at the quiet rate)
can still be warned through its flood risk. High is a cut-off on p_flood among warned ticks. On the development data no
cut-off reached the 1-in-3 flood precision Round 1 asks of High, so no fit ever says High.

Why the logistic flood rung works and the LightGBM one doesn't: 179 flood rows from about 15 storms. The trees overfit
(3 floods). The penalized regression (L2, non-negative coefficients on the rain and forecast inputs, penalty chosen out
of fold) flags 5. In the hybrid its weight goes to the hour (20%), 15-minute rain at the spot (18%), the forecast (12%)
and rain upwind (11%). That is the signal that made the rule win on floods, plus the timing a rule lacks. Its penalty search chose the strongest value tried (100), and the validation
loss is nearly flat across penalties. So the flood rung is barely constrained by the data, and 13 judged floods can't
say more.

### No gate, tested: the gate stays

Every recorded row trains both rungs. Quiet rows are sampled in two strata and weighted back up by 1/share:

| stratum | share of rows | holds | sampled at |
|---|---|---|---|
| stormy (rain within 20 km in the hour, or rain forecast) | 31% | 97.7% of alert rows, 98.3% of flood rows | 20% |
| dry (the rest) | 69% | the remainder | 2% |

The same numbers show that a rule-based gate can't reach 100% recall: even this loosest rule passes 31% of rows and
loses 2.3% of alert rows and 3 of 179 flood rows. With trees, though, going without a gate cost 8–10 points of events.
The learned gate passes 5.2% of rows to the rater, which then learns which storms bring alerts rather than whether
it's raining. The logistic rungs don't need the gate (26.1% without one): a smooth, heavily penalized model doesn't
spend itself on the bulk of dry rows the way boosting does.

### Features

| change | evidence | where |
|---|---|---|
| `dew_point_c` | Over the 289 ticks of 10 Nov 2025, relative humidity correlated −0.986 with temperature, and the dew point 0.08. Computed per station (Magnus formula) before the island mean. | `features.tick_weather` |
| `rain_prior_{3,6,24}h_mm` | The rain of the 2, 5 and 23 hours before the last one: with `rain_60min_mm`, the 3-, 6- and 24-hour totals, split so the burst at hand isn't counted twice. Point in time by construction: per gauge, a sum of its hourly totals each as published by its own tick. Checked against hand sums on 7,875 gauge-ticks. | `features.prior_rain` |
| wind as u/v | already so since 16 Oct, so 359° and 1° were already neighbours | – |
| rain vs the forecast | They overlap by design and split by rung. The alert rung leans on the forecast (41% of gain) and the rain 5–20 km away (19%), which give lead time before rain falls. The flood rung leans on 15-minute rain at the spot (30%). | `importance.json` per rung |

Both new inputs are optional (a null costs no training row) and point in time. They are computed once in `features.py`
for both gold and live scoring, and live and gold agreed on every column of 534 area-ticks. They live in the
`extended` feature set; `base` is v5's and the default. For the trees, extended cost events (30.1% → 25.5% for the
pair, 26.7% → 23.6% for the hybrid). For the logistic model the prior rain was worth +0.6 points and +0.5 minutes of
lead.

The antecedent rain runs against the usual intuition. Alert rows per 1,000 in `gold.model_rows`:

| last hour's rain | 2 h before: < 1 mm | 1–10 mm | 10–30 mm | 30+ mm |
|---|---|---|---|---|
| 1–10 mm | 2.5 | 1.0 | 0.7 | 0.26 |
| 10–30 mm | 7.0 | 3.7 | 2.3 | 3.2 |
| 30+ mm | 16.1 | 11.9 | 10.4 | 7.2 |

So the prior rain has no monotone constraint: the chance may fall with it.

**A data flag:** 104 training rows on 2 days, in Bedok and Tampines, show more than 512 mm in the 23 hours before the
last one. That is Singapore's daily record, so a gauge is likely misreporting there. Trees are robust to it, but silver
should be checked.

### Imbalance

- Quiet rows are subsampled and reweighted, as before.
- `flood_weight` (a flood row against a near miss) cost events at 3 here, as it did at 10 on 22 Oct, and flagged no
  more floods. Floods count 10 in *choosing* (the rank and the Medium cut-off), and 1 in training.
- The isotonic calibration of the alert rung and the Platt calibration of the flood rung are fitted on unweighted
  out-of-fold scores, so the chances stay probabilities.

### Evaluation

- **Time-blocked cross-validation, kept.** Five blocks of whole SGT days, balanced on alert days, with a one-day embargo
  either side. Every out-of-fold score comes from models and priors fitted without its block. Leave-one-event-out would
  mean 200+ folds for the same protection.
- **No separate validation split.** Blocked CV is a train/validation scheme that uses every event once for validation,
  and the final models are refitted on all of it. A single 20% validation block would hold about 3 of the 13 judged
  floods.
- **Measures:** events and floods flagged before PUB with Wilson intervals, lead and public warning with day-bootstrap
  intervals, false alarms per area-month, and, new, row-level PR-AUC and recall for alerts and floods, in every run.
- **The test:** the 2026 test was spent on 19 Oct, so it is no fresh measure. `flood-verify` still lists every 2026
  flood per model. The days from 13 Oct 2026 are the next clean look.

## Considered and not done

- **TabPFN.** TabPFN-3.5 exists (15 Sep 2026), but the Apache 2.0 licence covers the code only. Three things rule it
  out here:
  - it needs a GPU ("slow to execute on a CPU"), and Free Edition has none;
  - it is recommended up to 1M rows, and training has 13.9M, every one scored out of fold, plus 47 areas live every 5
    minutes;
  - its weights are non-commercial, barring production use of their outputs, which the resident app is.

  It could be an offline benchmark on a sample, on a GPU, with a Prior Labs token.
- **Bayesian hierarchical ordinal regression.** The location random intercept already exists: the Bühlmann–Straub
  credibility prior is its linear-Bayes form, and areas borrow strength through it. MCMC on 13.9M rows doesn't fit Free
  Edition. The missing piece is a random *slope*, the area effect weakening in heavy rain, which Lim Chu Kang (30 Sep
  2026) needs. Fitting one on case-control-sampled data is the next step.
- **CatBoost.** Its edge, ordered target statistics for categoricals, no longer applies: the area is an offset now. It
  would also add a dependency to four environments days before Demo Day. The small-data question it addresses was
  answered more cheaply: penalized regression did better than trees where data is thinnest.

## On Databricks

- **Compute (Free Edition, one serverless compute at a time).**
  - Gold rebuilt with the new columns in about 5 minutes of running.
  - Four training runs fitted 14 models, at 2–7 minutes a fit: about 2 hours of serverless in all, gold included.
  - The first sweep ran the driver out of memory: every fit copied the 13.9M development rows. The platform retried it
    from scratch, and that retry was cancelled.
  - Fits now share one sorted copy, the logistic learner works in float32, and the design frame is freed before the
    cut-off search.
- **The pipeline.**
  - Features stay in gold, defined once in `features.py` for the `flood-gold` pipeline and `flood-live`'s score task.
  - `flood-train` fits any mix of models in one MLflow run, each spec a nested run, with `sweep.json` comparing them.
    It registers the best as `@challenger` in Unity Catalog.
  - `flood-verify` checks the artifact before `@champion` moves. Live scoring loads `@champion` as a pyfunc and now
    passes it exactly the columns its signature declares.
- **Serving either kind.** The ordinal model is logged as `FloodSenseModel`, with its files in the `model` artifact. A
  pair is logged as `GatedPairModel` in `pair`, as v1–v5 were. `ml/models.py` loads either. The app uses the model's
  own logged loader and falls back to the pair's for versions before it.
- **A CLI trap.** `databricks bundle run --params` splits any value holding `=` on its commas. Quote the whole pair:

```bash
# v5's design, the hybrid and the logistic model side by side (base features unless :extended)
databricks bundle run train -t dev -p flood_dev --only train \
  --params '"models=gated,ordinal:alert=gated:flood=logistic,ordinal:alert=logistic:flood=logistic:extended"' \
  --params promote=false
databricks bundle run verify -t dev -p flood_dev                          # checks @challenger beside @champion
databricks bundle run verify -t dev -p flood_dev --params promote=true    # moves @champion if every check passes
```

Spec options:

- `ordinal` takes `alert=`, which is `lightgbm`, `logistic` or `gated`;
- `ordinal` takes `flood=`, which is `lightgbm` or `logistic`;
- `gated` takes `gate_recall=` and `area_effect=`;
- both take `flood_weight=`, `drop=<group>[+<group>]` and `extended`.

`--override false` skips the override search, and `compare_without` refits the winner without a feature group.

## Decisions

Made on 23 October, after the review above:

1. **v5 is `@champion`; the hybrid (v8) stays `@challenger`.** The user first promoted the hybrid (for its flood
   results), then switched back to v5 the same evening. Both are registered, and either can be `@champion` in a minute:
   `databricks registered-models set-alias <model> champion <5 or 8>`, one `flood-live` run so the map has a tick from
   the model, and an app restart (it caches the loaded model per process, and the track record for an hour). The
   hybrid passed all seven `flood-verify` checks, including a promoting run, so it can be promoted again as is.
2. **The false-alarm budget** stays at the rule's rate (0.87 per area-month). Raising it is still the biggest lever on
   recall: at 2 false alarms every design flags about 45% of events.
3. **Single-post id gaps** stay missing, not recorded. Reading them as recorded would add about 10 training floods.
4. **The Bedok and Tampines gauge readings** (the 512 mm flag above) are left as they are.
5. **24 Oct: v10 is `@champion`** (`gated:add=storm+surplus`, seed 0): v5's design plus the storm age and the rain surplus at 10–100 mm/h, chosen by the user. Against v5's design at three seeds each, its ROC AUC rose at every seed (+0.0009 on average) and it flagged as many floods (13 against 13), but slightly fewer PUB events (273 against 275). It passed all seven `flood-verify` checks. On the 5 judged 2026 floods no model trained on, it warned of none (v5: 1, the rule: 5). DEVELOPER_NOTES.md (Results so far, 24 Oct) has the sweeps. To go back: `databricks registered-models set-alias <model> champion 5`, one `flood-live` run, an app restart.

6. **24 Oct: the tiers by chance (v11), tried and set aside.** v11 has Low below 1%, Medium from 1% and High from 5%. v10's tiers are chosen when it is fitted: Medium within the rule's false-alarm rate, and High only where 1 in 3 High warnings end in a flood. No cut-off reaches that (the best is 1 in 20), so v10's High never shows. In v11, Medium is an early heads-up: 81% of events and 11 of 13 floods before PUB, 40 minutes ahead, at 5.4 false alarms per area-month. High warns at about the rule's rate (0.77 against 0.87), and that is where v11 is compared with the rule: 26% of events against 18%. The user promoted v11 at 22:32 SGT and switched back to v10 about half an hour later. So v10, its tiers and the app's wording for them are live, and v11 stays `@challenger`, re-promotable with an alias move, one `flood-live` run and the app copy for its tiers.

7. **25 Oct: v12 is `@champion`** (`gated:add=storm+surplus+drain+rise`, seed 0), chosen by the user: v10's design plus the flow into the drains and its overflow, from a nonlinear reservoir per gauge (outflow rising with storage to the power 5/3, as an open channel's with its depth), and each flow's rise over its highest earlier in the storm. Against v10's design at three seeds each, its ROC AUC, alert PR-AUC (+12%) and log loss improved at every seed and it flagged one more flood (14 against 13), but fewer PUB events (265 against 273), so it missed the bar set before the test. Its registration reproduced the sweep's fit, and it passed all seven `flood-verify` checks. On the 5 judged 2026 floods no model trained on, it warned of none (v10: none, the rule: 5). DEVELOPER_NOTES.md (Results so far, 25 Oct afternoon) has the sweeps. To go back: `databricks registered-models set-alias <model> champion 10`, one `flood-live` run, an app restart.

8. **25 Oct (evening): final_1, v12 with the tiers by chance, was `@champion` for an hour.** final_1 is v13, v12's fit with fixed cut-offs: Low below 1%, Moderate from 1% and Elevated from 5% (Medium and High in the data). Moderate is the early heads-up: 79% of events and 10 of 13 floods before PUB, 36 minutes ahead on average, at 5.3 false alarms per area-month. Elevated is where final_1 meets the rule: 29% of events against 18%, at 0.91 false alarms per area-month against 0.87. That is about 4% more false alarms than the verdict allows, so it misses the verdict and `flood-verify` won't promote it. The user made it `@champion` by hand anyway (about 20:36 SGT), and the app with its copy for these tiers went up at 20:57. It passed all seven `flood-verify` checks. On the 5 judged 2026 floods no model trained on, it warned of 4, all before PUB (v12: none, the rule: 5). To go back: `databricks registered-models set-alias <model> champion 12`, one `flood-live` run, and the app's wording back (DEVELOPER_NOTES.md, Results so far, 25 Oct evening, says where the 19:40 app source is kept).

9. **25 Oct (night): final_2, Moderate from 3%, is `@champion`.** final_2 is v14: final_1 with Moderate from 3% instead of 1%, so Low is below 3%. At Moderate it flags 45% of events before PUB (final_1: 79%) and 5 of 13 floods (final_1: 10), at 1.8 false alarms per area-month (final_1: 5.3). Those 5 floods are the ones Elevated flags too. Elevated is unchanged, so final_2 misses the verdict on false alarms as final_1 does. It passed all seven `flood-verify` checks. On the 5 judged 2026 floods it warned of 2 before PUB (final_1: 4, the rule: 3). The user chose the name and made it `@champion` at 21:31 SGT; final_1 stays registered on v13. To go back to final_1: `databricks registered-models set-alias <model> champion 13`, one `flood-live` run, an app restart. The app's wording fits both, as it reads the cut-offs from the live model's run.

What the app shows now (final_2): at Moderate, 44.6% of PUB events flagged before PUB (144 of 323) and 7 of 13 floods warned, 5 of them before PUB; at Elevated, 29.4% of events. Both come from final_2's own training run.

## Files

- New: `ml/ordinal.py` (the ladder, the warning score, the override), `ml/logistic.py`, `ml/models.py`,
  `tests/test_ml_ordinal.py`, `tests/test_ml_logistic.py`.
- Features: `features.py` (dew point, prior rain), `ml/contract.py` (the columns, `BASE_FEATURES`,
  `EXTENDED_FEATURES`, the `antecedent` group), `pipeline/utilities/contract_tables.py`, and the gold pipeline's
  `features.py` and `weather.py` (docs, expectations). `ml/live.py` reads a day of rain and the signature's columns.
- Training and checks:
  - `ml/train.py`: model specs, the override on every fit, PR-AUC, memory.
  - `ml/gated.py`: the override, the shared stats and loader face, `development_rows` without copies.
  - `ml/credibility.py`: flood-day claims.
  - `ml/verify.py` and `ml/serving.py`: either kind.
  - `ml/compare.py`.
  - `resources/train.job.yml`: `models`, `override`; default `gated`.
  - `resources/ml.yml`.
- App: `app/backend/model.py` loads either kind.
