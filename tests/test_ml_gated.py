import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense import features
from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import floods, gated
from daisi_team_master_corgis_floodsense.ml.scoreboard import score, tick_labels
from daisi_team_master_corgis_floodsense.ml.split import make_split

# Two months of storms over four areas. PUB alerts only in the two flood-prone ones, and only after heavy storms; NEA
# forecasts thundery showers from 90 minutes before every storm, and a storm rains 5-20 km away half an hour before it
# reaches an area. Other storms only pass by, raining 5-20 km away and never on the area, as most do. The weather
# stations' readings stop on day WEATHER_DAYS, as if their backfill hadn't landed.
START = pd.Timestamp("2025-03-01")
DAYS = 60
WEATHER_DAYS = 40
TEST_START = START + pd.Timedelta(days=DAYS)
FLOOD_PRONE = ("Ang Mo Kio", "Bedok")
AREAS = (*FLOOD_PRONE, "Changi", "Dover")
FAST = gated.Settings(max_rounds=300, patience=30, cut_off_candidates=30)


def synthetic(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gold.model_rows and silver.area_alerts for the storms."""
    rng = np.random.default_rng(seed)
    ticks = pd.date_range(START, TEST_START, freq="5min", inclusive="left")
    minutes = np.arange(len(ticks)) * 5.0
    frames, alerts = [], []
    hour = np.asarray(ticks.hour + ticks.minute / 60)
    day_heat = np.sin(np.pi * (hour - 8) / 12).clip(0)
    for area in AREAS:
        rain = np.zeros(len(ticks))  # mm per 5-minute reading
        passing = np.zeros(
            len(ticks)
        )  # the same, 5-20 km away, of storms that miss the area
        forecast = np.zeros(len(ticks), dtype=int)
        for day in range(DAYS):
            first = day * 288 + int(rng.integers(150, 220))  # 12:30 to 18:20
            length = int(rng.integers(6, 18))
            peak = rng.uniform(2, 12)
            storm = peak * np.hanning(length + 2)[1:-1]
            kind = rng.random()
            if kind < 0.6:
                if kind < 0.3:
                    passing[first : first + length] += storm
                    forecast[max(0, first - 18) : first + length] = 4
                continue
            rain[first : first + length] += storm
            forecast[max(0, first - 18) : first + length] = 4
            if area in FLOOD_PRONE and peak > 6:
                at = ticks[first] + pd.Timedelta(minutes=int(rng.integers(20, 45)))
                alerts.append((area, at, C.WARNING))
                if peak > 9:
                    alerts.append((area, at + pd.Timedelta(minutes=15), C.FLASH_FLOOD))
        # each window ends READING_LAG_MINUTES (two readings) before its tick
        seen = pd.Series(rain).shift(2, fill_value=0.0)
        nearby = pd.Series(rain).shift(-6, fill_value=0.0) + pd.Series(passing).shift(
            2, fill_value=0.0
        )
        nearby = nearby.rolling(6, min_periods=1).sum()
        landed = np.arange(len(ticks)) < WEATHER_DAYS * 288
        frames.append(
            pd.DataFrame(
                {
                    "area": area,
                    "ts": ticks,
                    **{
                        C.rain_column(w): seen.rolling(w // 5, min_periods=1)
                        .sum()
                        .round(1)
                        for w in C.RAIN_WINDOWS_MINUTES
                    },
                    # the hours before the last one, as gold sums them; unknown until enough hours are in
                    **{
                        C.prior_rain_column(h): (
                            seen.rolling(12 * h).sum() - seen.rolling(12).sum()
                        ).round(1)
                        for h in C.PRIOR_RAIN_HOURS
                    },
                    # the storm so far and the drains' response, as gold computes them from the gauge's readings
                    # (features.storm_age, rain_surplus, gauge_drains), one gauge per area here
                    **storm_so_far(seen.to_numpy()),
                    "rain_gauges": 3,
                    "forecast_rain_class": forecast,
                    **{
                        name: np.asarray(grouping)[forecast]
                        for name, grouping in C.FORECAST_GROUPINGS.items()
                    },
                    "forecast_age_min": 10.0,
                    "rain_nearby_30min_mm": nearby.round(1),
                    "rain_upwind_30min_mm": (0.8 * nearby).round(1),
                    "air_temp_c": np.where(landed, 27 + 4 * day_heat, np.nan),
                    "rel_humidity_pct": np.where(landed, 85 - 15 * day_heat, np.nan),
                    "dew_point_c": np.where(
                        landed, C.dew_point_c(27 + 4 * day_heat, 85 - 15 * day_heat), np.nan
                    ),
                    "wind_u_kn": np.where(landed, -3.0, np.nan),
                    "wind_v_kn": np.where(landed, 2.0, np.nan),
                    # a semidiurnal tide (12.42 h), the same everywhere; storms don't follow it here
                    "tide_height_m": 1.7 + np.cos(2 * np.pi * minutes / 745.2),
                    "minutes_to_high_tide": (-minutes) % 745.2,
                    # the area's ground, the same at every tick (gold.area_terrain)
                    **ground(area),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    # as train.load_rows reads gold: the area as a category; silver's alerts keep it as text
    rows["area"] = pd.Categorical(rows["area"], categories=sorted(AREAS))
    alerts = pd.DataFrame(alerts, columns=["area", "alert_ts", "kind"]).astype(
        {"area": object}
    )
    # PUB's alerts posted for other areas, as gold counts them (here every other area is near, and an alert is posted
    # at its stamp)
    lag = pd.Timedelta(minutes=C.POST_LAG_MINUTES)
    every = np.sort(alerts["alert_ts"].to_numpy(dtype="datetime64[ns]"))
    for km, minutes in C.NEARBY_ALERT_WINDOWS:
        counts = np.zeros(len(rows), dtype=int)
        for area in AREAS:
            at = (rows["area"] == area).to_numpy()
            end = (rows.loc[at, "ts"] - lag).to_numpy(dtype="datetime64[ns]")
            start = end - np.timedelta64(minutes, "m")
            own = np.sort(alerts.loc[alerts["area"] == area, "alert_ts"].to_numpy(dtype="datetime64[ns]"))
            n = lambda times: np.searchsorted(times, end, "right") - np.searchsorted(times, start, "right")
            counts[at] = n(every) - n(own)
        rows[C.nearby_alerts_column(km, minutes)] = counts
    rows["alert_label"] = tick_labels(rows, alerts)
    rows["flood_label"] = tick_labels(rows, alerts, kinds=[C.FLASH_FLOOD])
    rows["split"] = "train"
    return rows, alerts


def ground(area: str) -> dict[str, float]:
    """contract.MAP_FEATURES of an area, one value each: the flood-prone areas lie low by the sea, under hills."""
    i = sorted(AREAS).index(area)
    prone = area in FLOOD_PRONE
    return {
        "high_ground_km": 0.8 if prone else 2.5 + i,
        "relief_m": 60.0 if prone else 20.0,
        "road_share": 0.05 + 0.01 * i,
        "coast_km": 0.6 if prone else 3.0 + i,
        "waterway_km": 0.3 + 0.1 * i,
        "green_share": 0.1 if prone else 0.3,
    }


def storm_so_far(seen: np.ndarray) -> dict[str, np.ndarray]:
    """contract.STORM_HISTORY of one gauge's readings at their ticks (mm per 5 minutes, the newest READING_LAG back):
    the storm's age from the tick its 15-minute window first had rain after an hour without, and the leaky bucket at
    each drain rate. The plain recursions that features.py computes with windows."""
    window = pd.Series(seen).rolling(C.STORM_WINDOW_MINUTES // 5, min_periods=1).sum().to_numpy()
    gap, cap = C.STORM_GAP_MINUTES // 5, C.STORM_MAX_MINUTES
    age = np.full(len(seen), np.nan)
    start = last_wet = None
    for i, wet in enumerate(window > 0):
        if wet:
            if last_wet is None or i - last_wet > gap:
                start = i
            last_wet = i
        if last_wet is not None and i - last_wet <= gap:
            age[i] = min(5.0 * (i - start), cap)
    out = {C.STORM_AGE: age}
    for rate in C.DRAIN_RATES_MM_H:
        bucket, level = np.zeros(len(seen)), 0.0
        for i, mm in enumerate(seen):
            level = max(0.0, level + mm - rate / 12)
            bucket[i] = level
        out[C.surplus_column(rate)] = bucket
    # the drains: features.gauge_drains over the gauge's ticks, and each flow's rise over its highest earlier in the
    # storm, as far back as the age's cap
    drains = features.gauge_drains(
        pd.DataFrame({"station_id": "S1", "tick_s": 300 * np.arange(len(seen)), "mm": seen})
    )
    reach = C.STORM_MAX_MINUTES // 5
    for k, flow_name, overflow_name, rise_name in zip(
        C.DRAIN_RESPONSE_MINUTES, C.DRAIN_FLOW, C.DRAIN_OVERFLOW, C.DRAIN_RISE
    ):
        flow = drains[f"d{k}"].to_numpy()
        out[flow_name], out[overflow_name] = flow, drains[f"o{k}"].to_numpy()
        rise = np.full(len(seen), np.nan)
        start = None
        for i in range(len(seen)):
            if np.isnan(age[i]):
                continue
            start = i - int(round(age[i] / 5)) if age[i] < cap else start
            earlier = flow[max(start, i - reach) : i]
            rise[i] = round(flow[i] - (earlier.max() if len(earlier) else 0.0), 1)
        out[rise_name] = rise
    return out


ROWS, ALERTS = synthetic()
SPLIT = make_split(
    ALERTS,
    folds=3,
    start=START,
    test_start=TEST_START,
    test_end=TEST_START + pd.Timedelta(days=7),
)


@pytest.fixture(scope="module")
def fitted():
    return gated.fit(ROWS, ALERTS, SPLIT, settings=FAST)


def test_the_gate_passes_the_recall_share_and_screens_out_quiet_ticks(fitted):
    _, report = fitted
    s = report.stats
    # the share is of the alert rows' weight (a flood row weighing flood_weight, 1 by default)
    assert s["gate_weighted_recall"] >= FAST.gate_recall
    assert s["gate_alert_row_recall"] >= FAST.gate_recall
    assert s["gate_pass_share"] < 0.25  # most ticks are dry under a fair forecast
    assert s["gate_alert_event_recall"] == 1.0
    assert s["rater_rows"] == report.passed.sum() < len(report.rows)


def test_out_of_fold_tiers_flag_storms_within_the_false_alarm_budget(fitted):
    pair, report = fitted
    board = score(
        report.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end
    ).metrics(ci=False)
    assert board["alert_flagged_before_pub"] >= 0.5
    assert board["false_alarms_per_area_month"] <= FAST.max_false_alarms
    if math.isfinite(pair.high_p):
        assert board["high_precision"] >= FAST.min_high_precision
    # the cut-offs are the curve's choice
    assert pair.medium_p in set(report.curve["medium_p"])
    assert 0 <= pair.quiet_p < pair.medium_p < 1


def test_out_of_fold_the_pair_warns_before_each_flood_with_lead_time(fitted):
    # every flood of the development period, with the lead the out-of-fold pair gave it: the lead time given that a
    # flood occurred, as the scoreboard's flood metrics count it
    _, report = fitted
    table = floods.flood_table(
        report.predictions(warned_only=False), ALERTS, start=SPLIT.start, end=SPLIT.dev_end
    )
    flash = ALERTS[(ALERTS["kind"] == C.FLASH_FLOOD) & (ALERTS["alert_ts"] < SPLIT.dev_end)]
    assert len(table) == len(flash) > 0  # one storm a day, so each flood is its own event
    warned = table["lead_min"].notna()
    assert warned.mean() >= 0.8
    assert (table.loc[warned, "lead_min"] > 0).all()
    board = score(report.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end).metrics(ci=False)
    assert floods.summary(table)["floods_mean_lead_min"] == pytest.approx(board["flood_mean_lead_min"])
    assert table["before_pub"].mean() == pytest.approx(board["flood_flagged_before_pub"])


def test_the_pair_gives_every_flood_a_tier_and_a_chance(fitted):
    # the tier and chance the saved pair shows at each flood and at their peak in the hour before it
    pair, _ = fitted
    table = floods.flood_table(pair.predict(ROWS), ALERTS, start=START, end=TEST_START)
    assert len(table) == (ALERTS["kind"] == C.FLASH_FLOOD).sum()
    assert (table["ticks_scored"] == 12).all()
    assert table["tier_at_flood"].isin(C.TIERS).all() and table["peak_tier"].isin(C.TIERS).all()
    assert table["p_alert_at_flood"].between(0, 1).all() and table["peak_p_alert"].between(0, 1).all()
    # a warning is in force at a flood exactly when a tick of its hour was Medium or High
    assert (table["lead_min"].notna() == table["peak_tier"].isin(["Medium", "High"])).all()
    assert (table.loc[table["peak_tier"] != "Low", "peak_p_alert"] >= pair.medium_p).all()


def test_the_rater_learns_which_areas_flood(fitted):
    pair, _ = fitted
    preds = pair.predict(ROWS)
    warned = preds[preds["tier"] != "Low"]
    # storms hit all four areas alike, so a model blind to the area would warn in the flood-prone two about half the
    # time. Rain 5-20 km away heralds a storm everywhere, and the regularised rater lets the heaviest through elsewhere
    # too (16 Oct), so the test is that most warnings land where floods happen, not all.
    assert warned["area"].isin(FLOOD_PRONE).mean() >= 0.8
    quiet = (
        ROWS["rain_60min_mm"].eq(0)
        & ROWS["forecast_rain_class"].eq(0)
        & ROWS["rain_nearby_30min_mm"].eq(0)
    )
    assert (preds.loc[quiet, "tier"] == "Low").all()
    assert (preds.loc[quiet, "p_alert"] == pair.quiet_p).all()


def test_predictions_are_ml_predictions_rows_with_reasons(fitted):
    pair, _ = fitted
    preds = pair.predict(ROWS)
    assert list(preds.columns) == list(C.PREDICTIONS_COLUMNS)
    warned = preds["tier"] != "Low"
    assert warned.any()
    assert preds.loc[~warned, "reason"].isna().all()
    reasons = preds.loc[warned, "reason"]
    assert reasons.str.contains("mm of rain|NEA forecasts|PUB has often alerted").all()
    assert reasons.str.contains("mm of rain").any()
    assert preds["p_alert"].between(0, 1).all()


def test_a_row_missing_a_feature_is_not_rated(fitted):
    pair, _ = fitted
    storm = ROWS[ROWS["alert_label"] == 1].head(3).copy()
    storm["forecast_rain_class"] = np.nan
    preds = pair.predict(storm)
    assert (preds["tier"] == "Low").all() and preds["p_alert"].isna().all()


def test_a_row_missing_only_optional_features_is_rated(fitted):
    pair, _ = fitted
    storm = ROWS[ROWS["alert_label"] == 1].head(3).copy()
    storm[list(C.OPTIONAL_FEATURES)] = np.nan  # a weather outage, no upwind in a calm
    preds = pair.predict(storm)
    assert preds["p_alert"].between(0, 1).all()
    # and the rows without weather (the last weeks) trained the models
    assert ROWS["air_temp_c"].isna().any()
    gated.check_recorded(ROWS)


def test_predict_ignores_gold_columns_the_models_dont_read(fitted):
    # gold.area_features rows carry the held-back lightning columns too; a scorer can pass them as they are
    pair, _ = fitted
    rows = ROWS.head(500)
    with_lightning = rows.assign(**{c: 99.0 for c in C.HELD_BACK_FEATURES})
    pd.testing.assert_frame_equal(pair.predict(with_lightning), pair.predict(rows))
    assert not set(C.HELD_BACK_FEATURES) & set(gated.INPUTS)


def test_risk_only_rises_with_rain_the_forecast_and_the_tide():
    assert gated.RISING <= set(gated.INPUTS)
    assert {
        *gated.RAIN,
        *gated.NEARBY_RAIN,
        *C.PUB_ALERTS,
        "forecast_rain_class",
        *C.FORECAST_GROUPED,
        "tide_height_m",
        *C.DRAIN_RESPONSE,
    } == gated.RISING
    # the storm so far may move the chance either way, as the prior rain may: PUB alerts as a storm starts
    assert not {*C.STORM_HISTORY, *C.PRIOR_RAIN} & gated.RISING
    # and so may the ground: nobody knows in which direction a canal nearby moves the chance
    assert not set(C.MAP_FEATURES) & gated.RISING


def test_a_pair_reads_the_ground_and_a_pair_saved_before_it_still_loads(fitted, tmp_path):
    # the ground is one value per area, the same at every tick, read beside v5's features
    grounded = dataclasses.replace(
        FAST, drop=tuple(c for c in C.MODEL_FEATURES if c not in {*C.BASE_FEATURES, *C.MAP_FEATURES})
    )
    pair, _ = gated.fit(ROWS, ALERTS, SPLIT, settings=grounded)
    assert set(C.MAP_FEATURES) <= set(pair.features)
    pair.save(tmp_path / "ground")
    rows = ROWS.head(2000)
    pd.testing.assert_frame_equal(gated.GatedPair.load(tmp_path / "ground").predict(rows), pair.predict(rows))
    # a pair fitted without the ground (v5's features, as v10 has its own) loads and scores as before
    before, _ = fitted
    assert not set(C.MAP_FEATURES) & set(before.inputs)
    before.save(tmp_path / "before")
    pd.testing.assert_frame_equal(gated.GatedPair.load(tmp_path / "before").predict(rows), before.predict(rows))


def test_reasons_name_rain_further_away():
    row = {
        "area": "Bedok",
        "rain_nearby_30min_mm": 14.2,
        "rain_upwind_30min_mm": np.float32(9.6),
    }
    assert gated._phrase("rain_nearby_30min_mm", row) == (
        "14 mm of rain 5-20 km away in the last 30 min"
    )
    assert gated._phrase("rain_upwind_30min_mm", row) == (
        "10 mm of rain upwind, 5-20 km away in the last 30 min"
    )
    assert (
        gated._phrase("rain_upwind_30min_mm", {"rain_upwind_30min_mm": np.nan}) is None
    )
    assert gated._phrase("air_temp_c", {"air_temp_c": 33.0}) is None
    # PUB's own alerts nearby, in words; none, or unknown, says nothing
    assert gated._phrase("pub_alerts_5km_30min", {"pub_alerts_5km_30min": 1.0}) == (
        "a PUB alert within 5 km in the last 30 min"
    )
    assert gated._phrase("pub_alerts_10km_60min", {"pub_alerts_10km_60min": 3.0}) == (
        "3 PUB alerts within 10 km in the last 60 min"
    )
    for value in (0.0, np.nan):
        assert gated._phrase("pub_alerts_5km_60min", {"pub_alerts_5km_60min": value}) is None


def test_reasons_say_when_the_storm_sets_a_new_high():
    # the drains' rise in words, once even when both rises raised the score; below 1 mm/h, or outside a storm, nothing
    row = {"drain_rise_10min_mmh": 12.4, "drain_rise_30min_mmh": 3.1, "rain_15min_mm": 18.0}
    assert gated._phrase("drain_rise_10min_mmh", row) == "rain here is the heaviest it has been in this storm"
    assert gated.reason_text(["drain_rise_10min_mmh", "drain_rise_30min_mmh", "rain_15min_mm"], row) == (
        "rain here is the heaviest it has been in this storm; 18 mm of rain within 5 km in the last 15 min"
    )
    for value in (0.4, -6.0, np.nan):
        assert gated._phrase("drain_rise_30min_mmh", {"drain_rise_30min_mmh": value}) is None
    # the flows and overflows make no reason: the trees barely read them
    assert gated._phrase("drain_flow_10min_mmh", {"drain_flow_10min_mmh": 80.0}) is None


def test_save_and_load_predict_the_same(fitted, tmp_path):
    pair, _ = fitted
    pair.save(tmp_path)
    again = gated.GatedPair.load(tmp_path)
    pd.testing.assert_frame_equal(again.predict(ROWS), pair.predict(ROWS))


def test_missing_labels_never_reach_the_models():
    blank = ROWS.copy()
    blank.loc[blank.index[100:110], "alert_label"] = (
        np.nan
    )  # a missing window that slipped through
    with pytest.raises(ValueError, match="missing label or feature"):
        gated.fit(blank, ALERTS, SPLIT, settings=FAST)


def test_only_the_development_period_trains():
    later = ROWS.assign(ts=ROWS["ts"] + pd.Timedelta(days=DAYS), split="test")
    rows = gated.development_rows(pd.concat([ROWS, later]), SPLIT)
    assert rows["ts"].max() < SPLIT.dev_end and rows["ts"].min() >= SPLIT.start
    assert rows["ts"].is_monotonic_increasing


def test_recall_threshold_keeps_the_share_of_alert_rows():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.05])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    assert gated.recall_threshold(scores, y, 0.8) == 0.2  # drops 1 of 5
    assert gated.recall_threshold(scores, y, 1.0) == 0.1
    with pytest.raises(ValueError, match="no alert rows"):
        gated.recall_threshold(scores, np.zeros(7), 0.9)


def test_recall_threshold_keeps_a_low_scoring_flood_row():
    # the lowest alert row is a flood row, weighing 10: dropping it would lose 10 of 14, far beyond 20%
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
    y = np.array([1, 1, 1, 1, 1, 0])
    weight = np.array([10.0, 1, 1, 1, 1, 1])
    assert gated.recall_threshold(scores, y, 0.8, weight=weight) == 0.1
    # a warning row at the bottom still goes, as without weights
    weight = np.array([1.0, 10, 1, 1, 1, 1])
    assert gated.recall_threshold(scores, y, 0.9, weight=weight) == 0.2


def test_weighted_flagged_counts_a_flood_flood_weight_times():
    m = {
        "alert_events": 10,
        "alert_flagged_before_pub": 0.2,
        "flood_events": 2,
        "flood_flagged_before_pub": 0.5,
    }
    # (10 x 0.2 + 9 x 2 x 0.5) / (10 + 9 x 2)
    assert gated.weighted_flagged(m, 10) == pytest.approx(11 / 28)
    assert gated.weighted_flagged(m, 1) == 0.2
    assert gated.weighted_flagged({**m, "flood_events": 0, "flood_flagged_before_pub": math.nan}, 10) == 0.2


def test_the_area_is_a_credibility_prior_not_an_input(fitted):
    pair, report = fitted
    assert gated.AREA not in pair.inputs and pair.area_offsets is not None
    offsets = pair.area_offsets
    assert set(offsets) == set(AREAS)
    # the two areas that never alerted sit below the two that flood, but at a finite prior
    assert max(offsets[a] for a in ("Changi", "Dover")) < min(offsets[a] for a in FLOOD_PRONE)
    assert all(np.isfinite(v) for v in offsets.values())
    prior = report.area_prior.set_index("area")
    assert (prior.loc[["Changi", "Dover"], "rate"] == 0).all()
    assert (prior.loc[["Changi", "Dover"], "prior_rate"] > 0).all()
    assert report.stats["area_prior_k_days"] > 0


def test_the_area_as_a_category_still_fits_as_before():
    pair, report = gated.fit(
        ROWS, ALERTS, SPLIT, settings=dataclasses.replace(FAST, area_effect=gated.CATEGORY)
    )
    assert gated.AREA in pair.inputs and pair.area_offsets is None
    assert report.area_prior is None
    warned = pair.predict(ROWS).query("tier != 'Low'")
    assert warned["area"].isin(FLOOD_PRONE).mean() >= 0.8


def test_each_folds_prior_is_set_without_its_own_block():
    # Bedok's alerts all moved into the first fold's block: that fold's prior has never seen Bedok alert
    first = SPLIT.folds[0]
    rows = ROWS.copy()
    bedok = rows["area"] == "Bedok"
    rows.loc[bedok & ~first.valid(rows["ts"]), ["alert_label", "flood_label"]] = 0
    folds, full = gated._priors(rows, SPLIT, FAST)
    assert full.offsets["Bedok"] > full.offsets["Changi"]  # the whole period has seen Bedok's alerts
    assert folds[0]["Bedok"] == pytest.approx(folds[0]["Changi"])  # its first fold hasn't
    assert folds[1]["Bedok"] > folds[1]["Changi"]


def test_flood_rows_weigh_flood_weight_in_training():
    heavy = dataclasses.replace(FAST, flood_weight=10.0)
    _, weighted = gated.fit(ROWS, ALERTS, SPLIT, settings=heavy)
    _, unweighted = gated.fit(ROWS, ALERTS, SPLIT, settings=FAST)
    flood = weighted.rows["flood_label"].to_numpy() == 1
    assert flood.any()
    # the same rows, but the flood rows rate higher once they weigh more, and the gate lets them through first
    assert weighted.p_alert[flood].mean() > unweighted.p_alert[flood].mean()
    assert weighted.stats["gate_flood_row_recall"] >= weighted.stats["gate_alert_row_recall"]
    assert weighted.stats["flood_weight"] == 10 and unweighted.stats["flood_weight"] == 1
    # a flood flagged counts flood_value times in choosing the cut-off either way
    assert weighted.stats["flood_value"] == unweighted.stats["flood_value"] == 10


def test_settings_refuse_an_unknown_area_effect_or_a_light_flood():
    with pytest.raises(ValueError, match="area_effect"):
        gated.Settings(area_effect="categorical")
    with pytest.raises(ValueError, match="flood_weight"):
        gated.Settings(flood_weight=0.5)
    with pytest.raises(ValueError, match="flood_value"):
        gated.Settings(flood_value=0)


def test_a_pair_saved_before_the_prior_loads_and_scores_as_it_did(tmp_path):
    """A registered pair from before 21 Oct: the area as a category, trees fitted without an init_score (so they carry
    LightGBM's boost_from_average start), and no area_offsets, gate_base or rater_base in its pair.json."""
    inputs = gated.inputs_without(area=True)
    X = gated.design(ROWS, tuple(sorted(AREAS)), inputs)
    y = ROWS["alert_label"].to_numpy()
    params = {"objective": "binary", "verbosity": -1, "num_leaves": 7}
    gate = lgb.train(params, lgb.Dataset(X, y), num_boost_round=20)
    rater = lgb.train(params, lgb.Dataset(X, y), num_boost_round=20)
    old = gated.GatedPair(
        areas=tuple(sorted(AREAS)),
        gate=gate,
        gate_threshold=0.002,
        rater=rater,
        calibration_x=np.array([0.0, 0.5, 1.0]),
        calibration_y=np.array([0.0, 0.3, 0.9]),
        quiet_p=0.0001,
        medium_p=0.05,
        inputs=inputs,
    )
    old.save(tmp_path)
    saved = json.loads((tmp_path / "pair.json").read_text())
    for key in ("area_offsets", "gate_base", "rater_base", "flood_weight"):
        del saved[key]
    (tmp_path / "pair.json").write_text(json.dumps(saved))
    again = gated.GatedPair.load(tmp_path)
    assert again.area_offsets is None and again.gate_base == again.rater_base == 0.0
    passed, p = again.p_alert(ROWS)
    # exactly LightGBM's own probabilities, as the pair scored before 21 Oct
    np.testing.assert_array_equal(passed, gate.predict(X) >= 0.002)
    np.testing.assert_allclose(
        p[passed], again.calibrate(rater.predict(X[passed])), rtol=0, atol=0
    )


def test_usable_alerts_are_those_the_rows_can_learn_from():
    rows = ROWS[ROWS["ts"] < START + pd.Timedelta(days=30)]
    usable = gated.usable_alerts(ALERTS, rows)
    assert len(usable) and (usable["alert_ts"] < START + pd.Timedelta(days=30)).all()
    assert len(gated.usable_alerts(ALERTS, ROWS)) == len(ALERTS)


def test_no_cut_off_warns_the_rows_the_gate_screens_out():
    # three screened-out rows at the floor, and passed rows calibrated down to it: a cut-off at the floor would warn
    # every row, as one endless warning that the scoreboard counts as no false alarm
    rows = pd.DataFrame(
        {
            "area": "Bedok",
            "ts": pd.date_range(START, periods=6, freq="5min"),
        }
    )
    p = np.array([0.01, 0.01, 0.01, 0.01, 0.3, 0.6])
    passed = np.array([False, False, False, True, True, True])
    _, _, curve = gated.cut_offs(rows, p, passed, ALERTS.iloc[:0], SPLIT, settings=FAST)
    assert (curve["medium_p"] > 0.01).all()


def test_centered_calibration_keeps_block_levels_and_orders_every_score():
    # isotonic blocks: 0.0 for scores 0.1-0.3, 0.2 for 0.4-0.6, 0.5 for 0.7-0.9
    x = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])
    y = np.array([0.0, 0.0, 0.2, 0.2, 0.5, 0.5])
    cx, cy = gated._centered(x, y)
    assert cx.tolist() == pytest.approx([0.2, 0.5, 0.8]) and cy.tolist() == [
        0.0,
        0.2,
        0.5,
    ]
    p = gated._calibrated(np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.95]), cx, cy, 0.0)
    assert p.tolist() == pytest.approx([0.0, 0.0, 0.1, 0.2, 0.35, 0.5])
    assert gated._calibrated(np.array([0.05]), cx, cy, 0.01).tolist() == [0.01]


def test_loading_a_pair_needs_no_scikit_learn():
    """train.log_model pins only lightgbm, numpy and pandas for the registered model, so importing gated (as
    serving.py does to load it) must work where scikit-learn isn't installed. A fresh interpreter, with sklearn
    unimportable, stands in for such an environment."""
    no_sklearn = (
        "import sys\n"
        "class NoSklearn:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'sklearn':\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, NoSklearn())\n"
        "from daisi_team_master_corgis_floodsense.ml.gated import GatedPair\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    loaded = subprocess.run(
        [sys.executable, "-c", no_sklearn],
        env={**os.environ, "PYTHONPATH": str(src)},
        capture_output=True,
        text=True,
        check=False,  # the assert below shows the child's traceback
    )
    assert loaded.returncode == 0, loaded.stderr


def test_development_rows_pass_their_own_output_on_uncopied():
    # a sweep fits many models on ten million rows: re-sorting them for every fit ran a serverless driver out of memory
    dev = gated.development_rows(ROWS, SPLIT)
    assert gated.development_rows(dev, SPLIT) is dev
    shuffled = dev.sample(frac=1.0, random_state=0)
    again = gated.development_rows(shuffled, SPLIT)
    assert again is not shuffled and again[["area", "ts"]].equals(dev[["area", "ts"]])
    # rows past the development period are left out, as before
    assert gated.development_rows(ROWS, SPLIT)["ts"].max() < SPLIT.dev_end


def test_the_rule_can_be_the_pairs_safety_override(fitted, tmp_path):
    from dataclasses import replace

    from daisi_team_master_corgis_floodsense.ml import ordinal
    from daisi_team_master_corgis_floodsense.ml.baseline import RainfallRule

    pair, report = fitted
    dev = gated.development_rows(ROWS, SPLIT)
    chosen, refit, table = ordinal.add_override(pair, report, dev, ALERTS, SPLIT, RainfallRule(15, 5.0), settings=FAST)
    assert isinstance(chosen, gated.GatedPair)
    # the pair as fitted is the first option, and whatever wins stays within the budget
    assert table["override_mm"].isna().iloc[0] and table["medium_cut"].iloc[0] == pair.medium_p
    board = score(refit.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end).metrics(ci=False)
    assert board["false_alarms_per_area_month"] <= FAST.max_false_alarms
    # with an override set, every rated row whose rain passes it is warned, in the rule's words where only it warned
    forced = replace(pair, override=(15, 5.0))
    preds = forced.predict(ROWS)
    heavy = ROWS["rain_15min_mm"] >= 5.0
    assert (preds.loc[heavy, "tier"] != "Low").all()
    only_rule = heavy & (preds["p_alert"] < pair.medium_p)
    assert only_rule.any() and preds.loc[only_rule, "reason"].str.contains("in the last 15 min").all()
    assert (preds.loc[~heavy, "tier"] == pair.predict(ROWS).loc[~heavy, "tier"]).all()
    # it is saved with the pair, and a pair saved before 23 Oct loads without one
    forced.save(tmp_path / "pair")
    assert gated.GatedPair.load(tmp_path / "pair").override == (15, 5.0)
    assert forced.logged_cut_offs()["pair_override_mm"] == 5.0
    old = json.loads((tmp_path / "pair" / "pair.json").read_text())
    del old["override"]
    (tmp_path / "pair" / "pair.json").write_text(json.dumps(old))
    assert gated.GatedPair.load(tmp_path / "pair").override is None


def test_the_tiers_can_be_fixed_chances():
    # 24 Oct: the live tiers are chances of a PUB alert within the hour, not cut-offs chosen by budget and precision
    settings = dataclasses.replace(FAST, medium_chance=0.02, high_chance=0.1)
    pair, report = gated.fit(ROWS, ALERTS, SPLIT, settings=settings)
    assert (pair.medium_p, pair.high_p) == (0.02, 0.1)
    assert (report.stats["medium_p"], report.stats["high_p"]) == (0.02, 0.1)
    p = report.p_alert
    assert ((report.tier == "High") == (p >= 0.1)).all()
    assert ((report.tier != "Low") == (p >= 0.02)).all()
    # the Medium curve is still traced, so the fixed tiers can be read against the budget's choice
    assert len(report.curve) and {"medium_p", "flagged", "false_alarms"} <= set(report.curve.columns)
    # a saved pair keeps them
    assert pair.tiers(np.array([0.01, 0.05, 0.2])).tolist() == ["Low", "Medium", "High"]


def test_fixed_chances_must_be_ordered_chances():
    with pytest.raises(ValueError, match="above medium_chance"):
        gated.Settings(medium_chance=0.05, high_chance=0.05)
    with pytest.raises(ValueError, match="between 0 and 1"):
        gated.Settings(medium_chance=1.5)
    assert gated.Settings().medium_chance is None  # chosen by the budget, as sweeps compare fits
