import math

import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.baseline import RainfallRule, fit

DAY = pd.Timestamp("2026-01-05")
RAIN = C.rain_column(30)


def at(day: int, hhmm: str) -> pd.Timestamp:
    hours, minutes = hhmm.split(":")
    return DAY + pd.Timedelta(days=day, hours=int(hours), minutes=int(minutes))


def storm(area: str, day: int, first: str, mm: list[float]) -> pd.DataFrame:
    """Consecutive ticks from `first` with these 30-minute rainfall totals."""
    ts = pd.date_range(at(day, first), periods=len(mm), freq="5min")
    return pd.DataFrame({"area": area, "ts": ts, RAIN: mm})


FEATURES = pd.concat(
    [
        storm("Bukit Timah", 0, "16:00", [5, 12, 24, 35, 42, 38, 20]),
        storm("Bedok", 2, "16:00", [8, 18, 21, 14]),  # no alert follows
        storm("Jurong West", 4, "10:00", [10, 26, 30]),
    ],
    ignore_index=True,
)
ALERTS = pd.DataFrame(
    [
        ("Bukit Timah", at(0, "16:46"), C.WARNING),
        ("Bukit Timah", at(0, "17:11"), C.FLASH_FLOOD),
        ("Jurong West", at(4, "10:40"), C.WARNING),
    ],
    columns=["area", "alert_ts", "kind"],
)
TRAINING = {"start": DAY, "end": DAY + pd.Timedelta(days=30), "n_areas": 3}
GRID = (10.0, 20.0, 22.5, 25.0, 30.0, 40.0)


def test_rule_tiers_and_reasons():
    features = storm("Bedok", 0, "10:00", [10, 20, 45, float("nan")])
    out = RainfallRule(30, medium_mm=20, high_mm=40).predict(features)
    assert out["tier"].tolist() == ["Low", "Medium", "High", "Low"]
    assert out["reason"].iloc[2] == "45 mm of rain within 5 km in the last 30 min"
    assert out["reason"].iloc[[0, 3]].isna().all()
    assert set(out.columns) == set(C.PREDICTIONS_COLUMNS)


def test_fit_takes_the_most_sensitive_threshold_within_the_false_alarm_budget():
    rule, curve = fit(
        FEATURES, ALERTS, **TRAINING, windows=(30,), thresholds=GRID, max_false_alarms=0
    )
    # 20 mm raises a false alarm in Bedok, and 25 mm warns 5 minutes later than 22.5 mm in Bukit Timah
    assert (rule.window_min, rule.medium_mm) == (30, 22.5)
    by_mm = curve.set_index("medium_mm")
    assert by_mm.loc[20.0, "false_alarms"] > 0
    assert by_mm.loc[22.5, "flagged"] == 1 and by_mm.loc[40.0, "flagged"] == 0.5
    assert by_mm.loc[22.5, "lead_min"] == (36 + 35) / 2
    assert by_mm.loc[22.5, "tick_share"] == pytest.approx(6 / (3 * 30 * 288))


def test_fit_puts_high_at_the_lowest_threshold_floods_follow_often_enough():
    rule, _ = fit(
        FEATURES, ALERTS, **TRAINING, windows=(30,), thresholds=GRID, max_false_alarms=0
    )
    assert rule.high_mm == 25  # Bukit Timah flooded, Jurong West didn't: 1 in 2
    strict, _ = fit(
        FEATURES,
        ALERTS,
        **TRAINING,
        windows=(30,),
        thresholds=GRID,
        max_false_alarms=0,
        min_high_precision=0.75,
    )
    assert strict.high_mm == 40  # only Bukit Timah reached 40 mm


def test_without_floods_the_rule_never_says_high():
    warnings_only = ALERTS[ALERTS["kind"] == C.WARNING]
    rule, _ = fit(
        FEATURES,
        warnings_only,
        **TRAINING,
        windows=(30,),
        thresholds=GRID,
        max_false_alarms=0,
    )
    assert rule.high_mm == math.inf


def test_fit_refuses_when_no_candidate_fits_the_budget():
    with pytest.raises(ValueError, match="false alarms"):
        fit(
            FEATURES,
            ALERTS,
            **TRAINING,
            windows=(30,),
            thresholds=(10.0, 20.0),
            max_false_alarms=0,
        )
