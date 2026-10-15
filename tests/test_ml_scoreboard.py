import math

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.scoreboard import (
    MONTH,
    alert_events,
    score,
    tick_labels,
    warning_episodes,
    wilson,
)

DAY = pd.Timestamp("2026-09-22")
WEEK = {"start": DAY, "end": DAY + pd.Timedelta(days=7)}


def at(hhmm: str, day: int = 0) -> pd.Timestamp:
    hours, minutes = hhmm.split(":")
    return DAY + pd.Timedelta(days=day, hours=int(hours), minutes=int(minutes))


def alerts(*rows: tuple) -> pd.DataFrame:
    """(area, "HH:MM", kind) or (area, "HH:MM", kind, day) rows."""
    return pd.DataFrame(
        [(area, at(hhmm, *day), kind) for area, hhmm, kind, *day in rows],
        columns=["area", "alert_ts", "kind"],
    )


def warned(area: str, first: str, last: str, tier: str = "Medium", day: int = 0):
    """Every tick from first to last at `tier`."""
    ts = pd.date_range(at(first, day), at(last, day), freq="5min")
    return pd.DataFrame({"area": area, "ts": ts, "tier": tier})


# The 22 Sep 2026 King's Road flood: PUB's 90% warning at 16:46, the flood at 17:11.
KINGS_ROAD = alerts(
    ("Bukit Timah", "16:46", C.WARNING), ("Bukit Timah", "17:11", C.FLASH_FLOOD)
)


def test_alerts_at_most_an_hour_apart_are_one_event():
    events = alert_events(
        alerts(
            ("Bukit Timah", "16:46", C.WARNING),
            ("Bukit Timah", "16:50", C.WARNING),
            ("Bukit Timah", "17:11", C.FLASH_FLOOD),
            ("Bukit Timah", "18:30", C.WARNING),  # 79 min after the last: a new event
            ("Bedok", "16:50", C.WARNING),
        )
    )
    assert len(events) == 3
    storm = events[events["n_alerts"] == 3].iloc[0]
    assert storm["first_alert"] == at("16:46") and storm["first_flood"] == at("17:11")
    assert events["first_flood"].isna().sum() == 2


def test_warned_ticks_at_most_an_hour_apart_are_one_episode():
    episodes = warning_episodes(
        pd.concat(
            [
                warned("Ang Mo Kio", "10:00", "10:05"),
                warned("Ang Mo Kio", "11:05", "11:05"),  # 60 min on: same episode
                warned("Ang Mo Kio", "12:10", "12:10", "High"),  # 65 min on: a new one
                warned("Bedok", "10:00", "10:30", "Low"),
            ]
        )
    )
    assert episodes["start"].tolist() == [at("10:00"), at("12:10")]
    assert episodes["until"].tolist() == [at("12:05"), at("13:10")]
    assert episodes["ticks"].tolist() == [3, 1]


def test_a_flood_flagged_before_pubs_first_alert():
    board = score(
        warned("Bukit Timah", "16:20", "16:50"), KINGS_ROAD, **WEEK, n_areas=1
    )
    flood = board.events[board.events["kind"] == "flood"].iloc[0]
    assert flood["lead_min"] == 51 and flood["pub_lead_min"] == 25
    assert flood["before_pub"]

    m = board.metrics(ci=False)
    assert m["alert_flagged_before_pub"] == 1 and m["alert_mean_lead_min"] == 26
    assert m["flood_flagged_before_pub"] == 1
    assert m["flood_mean_warning_min"] == 51 and m["flood_mean_pub_warning_min"] == 25
    assert m["false_alarms_per_area_month"] == 0


def test_a_warning_after_pubs_first_alert_adds_nothing():
    board = score(
        warned("Bukit Timah", "16:50", "17:00"), KINGS_ROAD, **WEEK, n_areas=1
    )
    m = board.metrics(ci=False)
    assert m["alert_flagged_before_pub"] == 0  # the warning at 16:46 came first
    assert m["flood_mean_lead_min"] == 21 and m["flood_flagged_before_pub"] == 0
    assert m["flood_mean_warning_min"] == 25  # PUB's own 25 minutes stand


def test_an_expired_or_simultaneous_warning_is_a_miss():
    preds = pd.concat(
        [
            warned("Bukit Timah", "14:00", "14:00"),  # in force until 15:00
            warned("Bukit Timah", "16:45", "16:45"),
        ]
    )
    board = score(preds, alerts(("Bukit Timah", "16:45", C.WARNING)), **WEEK, n_areas=1)
    assert board.events["warned_from"].isna().all()
    assert not board.episodes["hit"].any()  # neither saw an alert after it started


def test_an_alert_confirms_a_warning_until_an_hour_after_its_last_tick():
    preds = warned("Ang Mo Kio", "10:00", "10:00")
    on_time = score(preds, alerts(("Ang Mo Kio", "11:00", C.WARNING)), **WEEK)
    too_late = score(preds, alerts(("Ang Mo Kio", "11:01", C.WARNING)), **WEEK)
    assert on_time.episodes["hit"].all() and not too_late.episodes["hit"].any()


def test_false_alarms_per_area_month_and_high_precision():
    preds = pd.concat(
        [
            warned("Ang Mo Kio", "10:30", "10:40", "High"),  # then a 90% warning only
            warned("Bedok", "10:30", "10:40", "High"),  # then a flash flood
            warned("Changi", "10:30", "10:40", "High"),  # then nothing
        ]
    )
    pub = alerts(("Ang Mo Kio", "11:00", C.WARNING), ("Bedok", "11:00", C.FLASH_FLOOD))
    m = score(preds, pub, start=DAY, end=DAY + MONTH, n_areas=3).metrics(ci=False)
    assert m["warnings"] == 3
    assert m["false_alarms_per_area_month"] == pytest.approx(1 / 3)
    assert m["warned_hours_per_area_month"] == pytest.approx(3 * (70 / 60) / 3)
    assert m["high_warnings"] == 3 and m["high_precision"] == pytest.approx(1 / 3)


def test_nothing_warned_flags_nothing_and_raises_no_alarms():
    nothing = warned("Bukit Timah", "16:00", "16:00").iloc[:0]
    m = score(nothing, KINGS_ROAD, **WEEK).metrics()
    assert m["alert_events"] == 1 and m["alert_flagged_before_pub"] == 0
    assert m["flood_mean_warning_min"] == 25  # PUB alone
    assert m["warnings"] == 0 and m["false_alarms_per_area_month"] == 0
    assert math.isnan(m["high_precision"])


def test_no_alerts_makes_every_warning_a_false_alarm():
    m = score(warned("Bedok", "10:00", "10:10"), KINGS_ROAD.iloc[:0], **WEEK).metrics()
    assert m["alert_events"] == 0 and math.isnan(m["alert_flagged_before_pub"])
    assert m["warnings"] == 1 and m["false_alarms_per_area_month"] > 0


def test_intervals_resample_whole_days():
    pub = alerts(
        ("Bukit Timah", "16:46", C.WARNING, 0),
        ("Bukit Timah", "17:11", C.FLASH_FLOOD, 0),
        ("Novena", "16:50", C.FLASH_FLOOD, 0),
        ("Clementi", "15:00", C.WARNING, 2),
        ("Clementi", "15:20", C.FLASH_FLOOD, 2),
    )
    preds = pd.concat(
        [
            warned("Bukit Timah", "16:20", "16:50"),
            warned("Clementi", "14:30", "15:00", day=2),
        ]
    )
    m = score(preds, pub, **WEEK).metrics(n_boot=500)
    assert m["flood_events"] == 3 and m["flood_days"] == 2
    assert m["flood_flagged_before_pub"] == pytest.approx(2 / 3)
    assert m["flood_mean_warning_min"] == pytest.approx((51 + 0 + 50) / 3)
    assert m["flood_mean_warning_min_lo"] <= m["flood_mean_warning_min"]
    assert m["flood_mean_warning_min"] <= m["flood_mean_warning_min_hi"]
    assert (
        m["flood_flagged_before_pub_lo"],
        m["flood_flagged_before_pub_hi"],
    ) == wilson(2, 3)


def test_wilson_interval():
    lo, hi = wilson(7, 8)
    assert lo == pytest.approx(0.529, abs=1e-3) and hi == pytest.approx(0.978, abs=1e-3)
    assert wilson(0, 10)[0] == pytest.approx(0) and wilson(10, 10)[1] == pytest.approx(
        1
    )
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_tick_labels_look_one_hour_ahead():
    ticks = pd.DataFrame(
        {
            "area": ["Ang Mo Kio"] * 5 + ["Bedok"],
            "ts": [
                at(t) for t in ("09:25", "09:30", "10:25", "10:30", "10:35", "10:00")
            ],
        }
    )
    pub = alerts(("Ang Mo Kio", "10:30", C.WARNING))
    assert tick_labels(ticks, pub).tolist() == [0, 1, 1, 0, 0, 0]
    assert tick_labels(ticks, pub, kinds=[C.FLASH_FLOOD]).sum() == 0


def test_an_episode_is_a_hit_exactly_when_one_of_its_ticks_is_labelled():
    rng = np.random.default_rng(7)
    grid = pd.date_range(DAY, DAY + pd.Timedelta(days=2), freq="5min", inclusive="left")
    preds = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], 150),
            "ts": rng.choice(grid, 150),
            "tier": "Medium",
        }
    ).drop_duplicates(["area", "ts"])
    pub = pd.DataFrame(
        {
            "area": rng.choice(["A", "B"], 30),
            "alert_ts": DAY
            + pd.to_timedelta(rng.integers(0, 2 * 24 * 60, 30), unit="min"),
            "kind": C.WARNING,
        }
    )
    board = score(preds, pub, start=DAY, end=DAY + pd.Timedelta(days=2), n_areas=2)
    labels = tick_labels(preds, pub)
    episodes = board.episodes[board.episodes["tier"] == "Medium"]
    assert 5 < len(episodes) and 0 < episodes["hit"].sum() < len(episodes)
    for ep in episodes.itertuples():
        ticks = (preds["area"] == ep.area) & preds["ts"].between(ep.start, ep.last)
        assert ep.hit == bool(labels[ticks].any())


def gap(first: str, last: str, day: int = 0) -> pd.Series:
    """Ticks from first to last whose labels are missing."""
    return pd.Series(pd.date_range(at(first, day), at(last, day), freq="5min"))


def test_an_event_after_a_missing_hour_is_not_judged():
    # a warning from 16:00 flags King's Road; but if 15:50-16:40 is missing, PUB may have alerted before 16:46
    preds = warned("Bukit Timah", "16:00", "17:10")
    whole = score(preds, KINGS_ROAD, **WEEK).metrics(ci=False)
    assert whole["alert_events"] == 1 and whole["alert_events_unjudged"] == 0
    blind = score(preds, KINGS_ROAD, **WEEK, missing=gap("15:50", "16:40"))
    m = blind.metrics(ci=False)
    assert (m["alert_events"], m["alert_events_unjudged"]) == (0, 1)
    # the flood event's lead-up includes PUB's first alert, so it goes too
    assert (m["flood_events"], m["flood_events_unjudged"]) == (0, 1)
    # missing ticks after the event don't matter
    late = score(preds, KINGS_ROAD, **WEEK, missing=gap("17:15", "18:00"))
    assert late.metrics(ci=False)["alert_events"] == 1
    # nor do missing ticks before it whose windows go missing only after it: 15:45 and 15:50 cover the hour to 16:46
    after = score(preds, KINGS_ROAD, **WEEK, missing=gap("15:55", "16:45"))
    assert after.metrics(ci=False)["alert_events"] == 1
    # the tick an hour before is what decides
    hour = score(preds, KINGS_ROAD, **WEEK, missing=gap("15:45", "15:45"))
    assert hour.metrics(ci=False)["alert_events_unjudged"] == 1


def test_a_quiet_warning_is_a_false_alarm_only_if_its_whole_watch_was_recorded():
    preds = warned("Bedok", "12:00", "12:30", day=2)  # no alert follows
    assert score(preds, KINGS_ROAD, **WEEK).metrics(ci=False)["warnings"] == 1
    blind = score(preds, KINGS_ROAD, **WEEK, missing=gap("12:20", "12:20", day=2))
    m = blind.metrics(ci=False)
    assert (
        m["warnings"],
        m["warnings_unjudged"],
        m["false_alarms_per_area_month"],
    ) == (0, 1, 0)
    # a confirmed warning is a hit whatever else is missing
    hit = score(
        warned("Bukit Timah", "16:30", "16:40"),
        KINGS_ROAD,
        **WEEK,
        missing=gap("16:35", "16:35"),
    )
    assert hit.metrics(ci=False)["warnings"] == 1 and hit.episodes["hit"].all()


def test_missing_time_is_not_time_at_risk():
    preds = warned("Bedok", "12:00", "12:30", day=2)
    whole = score(preds, KINGS_ROAD, **WEEK, n_areas=1)
    # a missing day elsewhere in the week leaves 6 recorded days of exposure
    blind = score(
        preds, KINGS_ROAD, **WEEK, n_areas=1, missing=gap("00:00", "23:55", day=5)
    )
    assert whole.area_months == pytest.approx(7 / 30.4375)
    assert blind.area_months == pytest.approx(6 / 30.4375)
    assert blind.metrics(ci=False)["false_alarms_per_area_month"] == pytest.approx(
        30.4375 / 6
    )


def test_missing_ticks_may_be_aware():
    sgt = gap("15:50", "16:40").dt.tz_localize("Asia/Singapore").dt.tz_convert("UTC")
    m = score(
        warned("Bukit Timah", "16:00", "17:10"), KINGS_ROAD, **WEEK, missing=sgt
    ).metrics(ci=False)
    assert m["alert_events_unjudged"] == 1


def test_an_event_the_model_was_blind_to_is_not_judged():
    # King's Road: PUB's first alert at 16:46, so warned ticks from 15:46 could have flagged it in time
    preds = warned("Bukit Timah", "16:00", "17:10")
    blind = score(preds, KINGS_ROAD, **WEEK, blind=gap("16:20", "16:30")).metrics(
        ci=False
    )
    assert (blind["alert_events"], blind["alert_events_unjudged"]) == (0, 1)
    assert (blind["flood_events"], blind["flood_events_unjudged"]) == (0, 1)
    # the first tick that counts is 15:50 (in force until 16:50); 15:45 expires a minute before the alert
    assert (
        score(preds, KINGS_ROAD, **WEEK, blind=gap("15:50", "15:50")).metrics(ci=False)[
            "alert_events_unjudged"
        ]
        == 1
    )
    early = score(preds, KINGS_ROAD, **WEEK, blind=gap("15:00", "15:45"))
    assert early.metrics(ci=False)["alert_events"] == 1
    # blind after both events: both judged
    late = score(preds, KINGS_ROAD, **WEEK, blind=gap("17:15", "18:00")).metrics(
        ci=False
    )
    assert (late["alert_events"], late["flood_events"]) == (1, 1)
    # blind at 16:50, between them: a warned tick then could have kept a warning in force at the 17:11 flood
    mid = score(preds, KINGS_ROAD, **WEEK, blind=gap("16:50", "16:50")).metrics(
        ci=False
    )
    assert (mid["alert_events"], mid["flood_events_unjudged"]) == (1, 1)


def test_blind_warnings_still_count_and_blind_time_is_not_at_risk():
    preds = warned("Bedok", "12:00", "12:30", day=2)  # no alert follows
    # the model saw what it warned on, so a blind tick inside the watch doesn't excuse the false alarm
    m = score(preds, KINGS_ROAD, **WEEK, blind=gap("12:20", "12:20", day=2)).metrics(
        ci=False
    )
    assert (m["warnings"], m["warnings_unjudged"]) == (1, 0)
    day = gap("00:00", "23:55", day=5)
    blind = score(preds, KINGS_ROAD, **WEEK, n_areas=1, blind=day)
    assert blind.area_months == pytest.approx(6 / 30.4375)
    # a tick both missing and blind is out of the time at risk once
    both = score(preds, KINGS_ROAD, **WEEK, n_areas=1, missing=day, blind=day)
    assert both.area_months == pytest.approx(6 / 30.4375)


def test_areas_may_come_as_categories():
    # train.py loads gold rows with the area as a category; silver's alerts carry it as text
    preds = warned("Bukit Timah", "16:20", "16:50").astype({"area": "category"})
    m = score(preds, KINGS_ROAD.astype({"area": object}), **WEEK).metrics(ci=False)
    assert m["alert_flagged_before_pub"] == 1 and m["false_alarms_per_area_month"] == 0
    ticks = preds[["area", "ts"]]
    assert tick_labels(ticks, KINGS_ROAD.astype({"area": object})).sum() == len(ticks)
