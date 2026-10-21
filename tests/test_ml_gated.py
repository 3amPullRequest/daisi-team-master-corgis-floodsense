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
# forecasts thundery showers from 90 minutes before every storm.
START = pd.Timestamp("2025-03-01")
DAYS = 60
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
    for area in AREAS:
        rain = np.zeros(len(ticks))  # mm per 5-minute reading
        forecast = np.zeros(len(ticks), dtype=int)
        for day in range(DAYS):
            if rng.random() > 0.4:
                continue
            first = day * 288 + int(rng.integers(150, 220))  # 12:30 to 18:20
            length = int(rng.integers(6, 18))
            peak = rng.uniform(2, 12)
            rain[first : first + length] += peak * np.hanning(length + 2)[1:-1]
            forecast[max(0, first - 18) : first + length] = 4
            if area in FLOOD_PRONE and peak > 6:
                at = ticks[first] + pd.Timedelta(minutes=int(rng.integers(20, 45)))
                alerts.append((area, at, C.WARNING))
                if peak > 9:
                    alerts.append((area, at + pd.Timedelta(minutes=15), C.FLASH_FLOOD))
        # each window ends READING_LAG_MINUTES (two readings) before its tick
        seen = pd.Series(rain).shift(2, fill_value=0.0)
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
                    # a semidiurnal tide (12.42 h), the same everywhere; storms don't follow it here
                    "tide_height_m": 1.7 + np.cos(2 * np.pi * minutes / 745.2),
                    "minutes_to_high_tide": (-minutes) % 745.2,
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
    assert set(warned["area"]) <= set(FLOOD_PRONE)
    quiet = ROWS["rain_60min_mm"].eq(0) & ROWS["forecast_rain_class"].eq(0)
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
    assert (
        reasons.str.contains("mm of rain").any() and reasons.str.contains("NEA").any()
    )
    assert preds["p_alert"].between(0, 1).all()


def test_a_row_missing_a_feature_is_not_rated(fitted):
    pair, _ = fitted
    storm = ROWS[ROWS["alert_label"] == 1].head(3).copy()
    storm["forecast_rain_class"] = np.nan
    preds = pair.predict(storm)
    assert (preds["tier"] == "Low").all() and preds["p_alert"].isna().all()


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
