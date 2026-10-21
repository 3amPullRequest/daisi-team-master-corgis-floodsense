"""Live scoring (ml/live.py) and the watchdog (freshness.py), checked without Spark."""

import math

import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense import freshness
from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import live
from daisi_team_master_corgis_floodsense.ml.baseline import RainfallRule

T = pd.Timestamp


def test_the_first_run_scores_the_newest_tick_whose_rain_has_landed():
    # the 14:30 reading feeds the 14:40 tick first (READING_LAG); 14:42 is after it
    assert live.ticks_to_score(T("2026-10-21 14:42"), T("2026-10-21 14:30"), None) == (
        T("2026-10-21 14:40"),
        T("2026-10-21 14:40"),
    )


def test_a_missed_run_is_caught_up_and_nothing_is_scored_twice():
    span = live.ticks_to_score(
        T("2026-10-21 15:01"), T("2026-10-21 14:50"), T("2026-10-21 14:40")
    )
    assert span == (T("2026-10-21 14:45"), T("2026-10-21 15:00"))
    # nothing new has landed since the last run
    assert (
        live.ticks_to_score(
            T("2026-10-21 15:03"), T("2026-10-21 14:50"), T("2026-10-21 15:00")
        )
        is None
    )


def test_a_long_outage_scores_only_the_last_two_hours():
    first, last = live.ticks_to_score(
        T("2026-10-21 20:00"), T("2026-10-21 19:50"), T("2026-10-20 08:00")
    )
    assert last == T("2026-10-21 20:00")
    assert last - first == pd.Timedelta(minutes=live.CATCH_UP_MINUTES)


def test_no_tick_runs_ahead_of_the_clock_and_no_rain_means_nothing_to_score():
    # a reading stamped ahead of the clock still can't score a future tick
    span = live.ticks_to_score(T("2026-10-21 14:42"), T("2026-10-21 14:55"), None)
    assert span == (T("2026-10-21 14:40"), T("2026-10-21 14:40"))
    assert live.ticks_to_score(T("2026-10-21 14:42"), None, None) is None


@pytest.mark.parametrize(
    "rule",
    [RainfallRule(30, 12.5), RainfallRule(15, 7.5, 20.0), RainfallRule(60, 25.0)],
)
def test_the_rule_comes_back_from_its_logged_name(rule):
    assert live.rule_from_name(rule.name) == rule
    with pytest.raises(ValueError):
        live.rule_from_name("gated_lightgbm M>=0.1 H>=inf")


def test_scorer_rows_are_float64_with_every_model_feature():
    frame = pd.DataFrame(
        {
            "area": ["Bedok"],
            "ts": [T("2026-10-21 14:40")],
            **{c: [1] for c in C.MODEL_FEATURES},
            "extra": ["dropped"],
        }
    )
    rows = live.model_rows(frame)
    assert list(rows.columns) == ["area", "ts", *C.MODEL_FEATURES]
    assert all(rows[c].dtype == "float64" for c in C.MODEL_FEATURES)


def test_the_watchdog_names_what_went_stale():
    now = T("2026-10-21 15:00")
    fresh = {
        "tick": T("2026-10-21 14:56"),
        "core sources": T("2026-10-21 14:56"),
        "rain reading": T("2026-10-21 14:45"),
        "prediction": T("2026-10-21 14:55"),
    }
    assert freshness.problems(now, fresh) == []
    stale = fresh | {"prediction": T("2026-10-21 13:00"), "core sources": None}
    found = freshness.problems(now, stale)
    assert len(found) == 2
    assert any("prediction is 120 min old" in p for p in found)
    assert "no core sources at all" in found
    assert all(not math.isnan(v) for v in freshness.LIMITS_MIN.values())
