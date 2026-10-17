import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import gated
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
                    "rain_gauges": 3,
                    "forecast_rain_class": forecast,
                    "forecast_age_min": 10.0,
                    "rain_nearby_30min_mm": nearby.round(1),
                    "rain_upwind_30min_mm": (0.8 * nearby).round(1),
                    "air_temp_c": np.where(landed, 27 + 4 * day_heat, np.nan),
                    "rel_humidity_pct": np.where(landed, 85 - 15 * day_heat, np.nan),
                    "wind_u_kn": np.where(landed, -3.0, np.nan),
                    "wind_v_kn": np.where(landed, 2.0, np.nan),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    # as train.load_rows reads gold: the area as a category; silver's alerts keep it as text
    rows["area"] = pd.Categorical(rows["area"], categories=sorted(AREAS))
    alerts = pd.DataFrame(alerts, columns=["area", "alert_ts", "kind"]).astype(
        {"area": object}
    )
    rows["alert_label"] = tick_labels(rows, alerts)
    rows["flood_label"] = tick_labels(rows, alerts, kinds=[C.FLASH_FLOOD])
    rows["split"] = "train"
    return rows, alerts


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


def test_risk_only_rises_with_rain_and_the_forecast():
    assert gated.RISING <= set(gated.INPUTS)
    assert {*gated.RAIN, *gated.NEARBY_RAIN, "forecast_rain_class"} == gated.RISING


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
