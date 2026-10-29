"""publish: the scoreboard tables the agencies' dashboard and Genie space read, from scoreboards and the training run."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import floods, publish
from daisi_team_master_corgis_floodsense.ml.scoreboard import score

DAY = pd.Timestamp("2026-02-18")
WINDOW = {"start": DAY, "end": DAY + pd.Timedelta(days=1), "missing": None, "blind": None}


def at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY.date()} {hhmm}")


def predictions(area: str, p_by_tick: dict[str, float], medium: float = 0.03, high: float = 0.05):
    """Every tick of the day for `area`, at p 0.001 (Low) unless `p_by_tick` says otherwise."""
    ticks = pd.date_range(DAY, DAY + pd.Timedelta(days=1), freq="5min", inclusive="left")
    p = pd.Series(0.001, index=ticks)
    for hhmm, value in p_by_tick.items():
        p[at(hhmm)] = value
    tier = np.select([p >= high, p >= medium], ["High", "Medium"], "Low")
    return pd.DataFrame({"area": area, "ts": ticks, "tier": tier, "p_alert": p.to_numpy()})


def alerts(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame([(area, at(hhmm), kind) for area, hhmm, kind in rows], columns=["area", "alert_ts", "kind"])


def as_logged(frame: pd.DataFrame) -> pd.DataFrame:
    """A table as mlflow.log_table keeps it and verify._artifact_table reads it back: timestamps as text, NaT too."""
    return frame.apply(lambda col: col.astype(str) if pd.api.types.is_datetime64_any_dtype(col) else col)


# Bedok: PUB's warning at 14:25 and a flash flood at 14:40. FloodSense rises from 14:00 and is Elevated at 14:10; the
# rule fires at 14:30, after PUB. In Tampines FloodSense warns at 09:00 and nothing follows: a false alarm.
PUB = alerts(("Bedok", "14:25", C.WARNING), ("Bedok", "14:40", C.FLASH_FLOOD))
OURS = pd.concat(
    [
        predictions("Bedok", {"14:00": 0.035, "14:05": 0.04, "14:10": 0.06, "14:15": 0.045, "14:20": 0.04}),
        predictions("Tampines", {"09:00": 0.032}),
    ],
    ignore_index=True,
)
RULES = pd.concat(
    [predictions("Bedok", {"14:30": 0.04}), predictions("Tampines", {})], ignore_index=True
).assign(p_alert=np.nan)


def test_after_training_tables_score_floodsense_at_both_levels_and_the_rule_on_the_same_storms():
    keys = publish.flood_hours(PUB, WINDOW["start"], WINDOW["end"])
    tables = publish.after_training_tables(
        publish.kept(OURS, keys), publish.kept(RULES, keys), PUB, WINDOW, "final_2", "15-minute rain"
    )
    board = tables["scoreboard"].set_index(["model", "warning_level"])
    assert list(board.index) == [
        (publish.FLOODSENSE, publish.MODERATE),
        (publish.FLOODSENSE, publish.ELEVATED),
        (publish.RULE, publish.HEAVY_RAIN),
    ]
    assert set(tables["scoreboard"]["period"]) == {publish.AFTER_TRAINING}
    ours = board.loc[(publish.FLOODSENSE, publish.MODERATE)]
    assert ours["pub_alert_events"] == 1 and ours["alerts_warned_before_pub"] == 1.0
    assert ours["alert_mean_lead_min"] == 25  # 14:00 to PUB's 14:25
    assert ours["flash_floods"] == 1 and ours["flood_mean_lead_min"] == 40
    assert ours["pub_flood_warning_min"] == 15  # PUB's own warning: 14:25 to 14:40
    assert ours["warnings"] == 2 and ours["warnings_followed_by_alert"] == 0.5  # Bedok's came true, Tampines' didn't
    assert ours["period_start"] == DAY.date() and ours["period_end"] == DAY.date()
    elevated = board.loc[(publish.FLOODSENSE, publish.ELEVATED)]
    assert elevated["alert_mean_lead_min"] == 15 and elevated["warnings"] == 1  # 14:10 only, never in Tampines
    rule = board.loc[(publish.RULE, publish.HEAVY_RAIN)]
    assert rule["alerts_warned_before_pub"] == 0.0 and rule["model_version"] == "15-minute rain"


def test_after_training_floods_put_floodsense_and_the_rule_side_by_side_in_the_app_s_tier_names():
    keys = publish.flood_hours(PUB, WINDOW["start"], WINDOW["end"])
    row = publish.after_training_tables(
        publish.kept(OURS, keys), publish.kept(RULES, keys), PUB, WINDOW, "final_2", "rule"
    )["scoreboard_floods"].iloc[0]
    assert row["flood_time"] == at("14:40") and row["pub_first_alert"] == at("14:25")
    assert row["floodsense_lead_min"] == 40 and row["floodsense_before_pub"]
    assert row["floodsense_peak_tier"] == "Elevated" and row["floodsense_peak_chance"] == pytest.approx(0.06)
    assert row["floodsense_tier_at_flood"] == "Low"  # 14:35: the warning was in force, but the chance had fallen
    assert row["rule_lead_min"] == 10 and not row["rule_before_pub"]
    assert row["scored_by"] == "final_2" and row["period"] == publish.AFTER_TRAINING


def test_after_training_events_and_warnings_carry_who_warned_at_which_level():
    keys = publish.flood_hours(PUB, WINDOW["start"], WINDOW["end"])
    tables = publish.after_training_tables(
        publish.kept(OURS, keys), publish.kept(RULES, keys), PUB, WINDOW, "final_2", "rule"
    )
    events = tables["scoreboard_events"]
    assert set(events["event_kind"]) == {"pub_alert", "flash_flood"}
    assert len(events) == 6  # one alert event and one flood, for each of the three rows
    warnings = tables["scoreboard_warnings"]
    ours = warnings[warnings["warning_level"] == publish.MODERATE].set_index("area")
    assert ours.loc["Tampines", "false_alarm"] and not ours.loc["Bedok", "false_alarm"]
    assert ours.loc["Bedok", "warning_start"] == at("14:00")
    assert ours.loc["Bedok", "in_force_until"] == at("15:20")  # an hour after its last warned tick
    assert ours.loc["Bedok", "hours_in_force"] == pytest.approx(80 / 60)


def test_kept_holds_the_warned_ticks_and_the_hour_before_each_flood_only():
    # Bedok warned at 12:00, long before its flood; Tampines neither warned nor flooded
    preds = pd.concat([predictions("Bedok", {"12:00": 0.04}), predictions("Tampines", {})], ignore_index=True)
    rows = publish.kept(preds, publish.flood_hours(PUB, WINDOW["start"], WINDOW["end"]))
    assert set(rows["ts"]) == set(floods.hour_before(at("14:40"))) | {at("12:00")}  # 13:40 to 14:35, Low included
    assert set(rows["area"]) == {"Bedok"}
    assert list(rows.columns) == ["area", "ts", "tier", "p_alert"]


def test_flood_hours_skip_floods_outside_the_window():
    keys = publish.flood_hours(PUB, DAY + pd.Timedelta(days=1), DAY + pd.Timedelta(days=2))
    assert len(keys) == 0
    assert publish.kept(OURS, keys)["tier"].ne("Low").all()


def test_a_share_s_interval_never_shows_below_zero():
    row = publish.summary_row({"flood_flagged_before_pub_lo": -2.78e-17, "alert_flagged_before_pub_hi": 1.0000001})
    assert row["floods_warned_before_pub_lo"] == 0.0 and row["alerts_warned_before_pub_hi"] == 1.0
    assert math.isnan(row["floods_warned_before_pub"])


def test_development_rows_are_the_training_run_s_metrics_under_each_prefix():
    metrics = {
        "oof_alert_events": 323.0,
        "oof_alert_flagged_before_pub": 0.4458,
        "oof_alert_flagged_before_pub_lo": 0.3926,
        "oof_alert_mean_warning_min_lo": 9.1,
        "oof_flood_mean_pub_warning_min": 20.78,
        "oof_false_alarms_per_area_month": 1.772,
        "oof_medium_warning_alert_share": 0.054,
        "oof_high_alert_flagged_before_pub": 0.2941,
        "oof_high_warning_alert_share": 0.075,
        "rule_dev_alert_flagged_before_pub": 0.1827,
        "rule_dev_warnings": 1560.0,
    }
    rows = publish.development_summary(metrics, "final_2", "rule", date(2021, 10, 27), date(2025, 12, 19))
    rows = rows.set_index("warning_level")
    ours, high, rule = rows.loc[publish.MODERATE], rows.loc[publish.ELEVATED], rows.loc[publish.HEAVY_RAIN]
    assert ours["pub_alert_events"] == 323 and ours["alerts_warned_before_pub_lo"] == 0.3926
    assert ours["alert_mean_lead_min_lo"] == 9.1  # an alert event's warning is its lead
    assert ours["pub_flood_warning_min"] == 20.78 and ours["warnings_followed_by_alert"] == 0.054
    assert high["alerts_warned_before_pub"] == 0.2941 and high["warnings_followed_by_alert"] == 0.075
    assert rule["warnings"] == 1560 and rule["model_version"] == "rule"
    assert math.isnan(rule["warnings_followed_by_alert"])  # the training run doesn't log it for the rule
    assert math.isnan(rule["pub_alert_events"])  # not logged: missing, not zero
    assert set(rows["period"]) == {publish.DEVELOPMENT}


def test_development_floods_read_back_from_mlflow_tables_with_the_rule_from_the_verify_run():
    ours = floods.flood_table(OURS, PUB, **WINDOW)
    verify = as_logged(
        ours[["area", "flood_ts"]].assign(lead_min_rule=10.0, before_pub_rule=False, lead_min=99.0)
    )
    events = as_logged(score(OURS[OURS["tier"] != "Low"], PUB, **WINDOW).events)
    tables = publish.development_tables(
        {}, events, as_logged(ours), verify, "final_2", "rule", date(2021, 10, 27), date(2025, 12, 19)
    )
    row = tables["scoreboard_floods"].iloc[0]
    assert row["flood_time"] == at("14:40") and row["floodsense_warned_from"] == at("14:00")
    assert row["floodsense_lead_min"] == 40  # FloodSense's out-of-fold lead, not verify's in-sample one
    assert row["rule_lead_min"] == 10 and not row["rule_before_pub"]
    assert row["scored_by"] == publish.OUT_OF_FOLD and row["period"] == publish.DEVELOPMENT
    events = tables["scoreboard_events"]
    missed = events[events["warned_from"].isna()]
    assert missed.empty and set(events["model"]) == {publish.FLOODSENSE}
    assert events["event_time"].dtype == "datetime64[ns]"


def test_a_flood_no_warning_was_in_force_at_reads_back_as_null_not_text():
    table = floods.flood_table(predictions("Bedok", {}), PUB, **WINDOW)
    row = publish.flood_rows(as_logged(table), None, publish.DEVELOPMENT, publish.OUT_OF_FOLD).iloc[0]
    assert pd.isna(row["floodsense_warned_from"]) and math.isnan(row["floodsense_lead_min"])
    assert row["public_warning_min"] == 15  # PUB's own warning still reached the public
    assert math.isnan(row["rule_lead_min"]) and row["rule_before_pub"] is None


def test_month_spans_cover_the_window_without_gaps_and_cut_at_month_starts():
    spans = publish.month_spans("2025-12-20", "2026-03-04 10:05")
    assert spans[0] == (pd.Timestamp("2025-12-20"), pd.Timestamp("2026-01-01"))
    assert spans[-1] == (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-04 10:05"))
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert publish.month_spans("2026-01-01", "2026-01-01") == []


def test_every_table_lists_its_columns_with_a_type_and_a_comment():
    for columns, comment in publish.TABLES.values():
        assert comment
        for name, (kind, doc) in columns.items():
            assert kind in {"STRING", "DATE", "BIGINT", "DOUBLE", "BOOLEAN", "TIMESTAMP_NTZ"}, name
            assert doc, name
    assert set(publish.SUMMARY.values()) <= set(publish.SCOREBOARD_COLUMNS)
