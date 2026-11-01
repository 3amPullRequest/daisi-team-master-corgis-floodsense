"""publish: the scoreboard tables the agencies' dashboard, the Genie space and the app read, over the test period."""

import math

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import floods, publish

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
    return pd.DataFrame(
        [(area, at(hhmm), kind, f"{area} Road") for area, hhmm, kind in rows],
        columns=["area", "alert_ts", "kind", "location"],
    )


# Bedok: PUB's warning at 14:25 and flash floods at 14:40 and 15:05. FloodSense rises from 14:00 (in force until 15:20)
# and is Elevated at 14:10 only; the rule fires at 14:30. In Tampines FloodSense warns at 09:00 and nothing follows.
PUB = alerts(("Bedok", "14:25", C.WARNING), ("Bedok", "14:40", C.FLASH_FLOOD), ("Bedok", "15:05", C.FLASH_FLOOD))
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


def tables():
    keys = publish.flood_hours(PUB, WINDOW["start"], WINDOW["end"])
    return publish.after_training_tables(
        publish.kept(OURS, keys), publish.kept(RULES, keys), PUB, WINDOW, "final_2", "15-minute rain"
    )


def test_the_scoreboard_scores_floodsense_at_both_levels_and_the_rule_on_the_same_floods():
    board = tables()["scoreboard"].set_index(["model", "warning_level"])
    assert list(board.index) == [
        (publish.FLOODSENSE, publish.MODERATE),
        (publish.FLOODSENSE, publish.ELEVATED),
        (publish.RULE, publish.HEAVY_RAIN),
    ]
    assert set(board["period"]) == {publish.AFTER_TRAINING}
    ours = board.loc[(publish.FLOODSENSE, publish.MODERATE)]
    assert ours["flash_floods"] == 2 and ours["floods_warned"] == 2
    # 14:40 came 40 minutes after the warning began; 15:05, 65 minutes after: past the hour its first rating warned of
    assert ours["floods_caught"] == 1 and ours["floods_caught_share"] == 0.5
    assert ours["flood_mean_lead_min"] == pytest.approx((40 + 65) / 2)  # over the floods a warning was in force at
    assert ours["warnings"] == 2  # Bedok's came true, Tampines' didn't
    assert ours["warnings_followed_by_alert"] == 0.5 and ours["warnings_followed_by_flood"] == 0.5
    assert ours["warnings_followed_by_alert_lo"] < 0.5 < ours["warnings_followed_by_alert_hi"]
    assert ours["period_start"] == DAY.date() and ours["period_end"] == DAY.date()
    elevated = board.loc[(publish.FLOODSENSE, publish.ELEVATED)]
    # 14:10 only, in force until 15:10: 30 and 55 minutes ahead, both caught; never in Tampines
    assert elevated["floods_caught"] == 2 and elevated["flood_mean_lead_min"] == pytest.approx(42.5)
    assert elevated["warnings"] == 1 and elevated["warnings_followed_by_flood"] == 1.0
    rule = board.loc[(publish.RULE, publish.HEAVY_RAIN)]
    assert rule["floods_caught"] == 2 and rule["flood_mean_lead_min"] == pytest.approx((10 + 35) / 2)
    assert rule["model_version"] == "15-minute rain"
    assert set(board.reset_index().columns) == set(publish.SCOREBOARD_COLUMNS)


def test_floods_put_floodsense_and_the_rule_side_by_side_one_row_per_report():
    floods_table = tables()["scoreboard_floods"]
    assert list(floods_table["flood_time"]) == [at("14:40"), at("15:05")]
    first, second = floods_table.iloc[0], floods_table.iloc[1]
    assert first["floodsense_warned_from"] == at("14:00") and first["floodsense_lead_min"] == 40
    assert first["floodsense_caught"] and not second["floodsense_caught"] and second["floodsense_lead_min"] == 65
    assert first["floodsense_peak_tier"] == "Elevated" and first["floodsense_peak_chance"] == pytest.approx(0.06)
    assert first["floodsense_tier_at_flood"] == "Low"  # 14:35: the warning was in force, but the chance had fallen
    assert first["rule_lead_min"] == 10 and first["rule_caught"] and second["rule_lead_min"] == 35
    assert first["location"] == "Bedok Road" and first["judged"]
    assert first["scored_by"] == "final_2" and first["period"] == publish.AFTER_TRAINING
    assert list(floods_table.columns) == list(publish.FLOODS_COLUMNS)


def test_warnings_carry_whether_a_pub_alert_or_a_flash_flood_followed():
    warnings = tables()["scoreboard_warnings"]
    ours = warnings[warnings["warning_level"] == publish.MODERATE].set_index("area")
    assert ours.loc["Tampines", "false_alarm"] and not ours.loc["Tampines", "followed_by_flash_flood"]
    assert ours.loc["Bedok", "followed_by_pub_alert"] and ours.loc["Bedok", "followed_by_flash_flood"]
    assert ours.loc["Bedok", "warning_start"] == at("14:00")
    assert ours.loc["Bedok", "in_force_until"] == at("15:20")  # an hour after its last warned tick
    assert ours.loc["Bedok", "hours_in_force"] == pytest.approx(80 / 60)
    assert set(warnings.columns) == set(publish.WARNINGS_COLUMNS)


def test_a_warning_a_drain_alert_alone_followed_is_no_flood_warning():
    pub = alerts(("Bedok", "14:25", C.WARNING))
    rows = publish.warning_rows(
        publish.score(OURS[OURS["tier"] != "Low"], pub, **WINDOW).episodes, pub
    ).set_index("area")
    assert rows.loc["Bedok", "followed_by_pub_alert"] and not rows.loc["Bedok", "followed_by_flash_flood"]


def test_a_share_s_interval_never_shows_below_zero():
    warnings = pd.DataFrame(
        {"judged": [True] * 3, "followed_by_pub_alert": [False] * 3, "followed_by_flash_flood": [False] * 3}
    )
    reports = floods.report_table(predictions("Bedok", {}), PUB, start=WINDOW["start"], end=WINDOW["end"])
    row = publish.summary_row(warnings, reports)
    assert row["warnings_followed_by_flood"] == 0.0 and row["warnings_followed_by_flood_lo"] == 0.0
    assert row["floods_caught_share_lo"] == 0.0 and row["floods_caught"] == 0
    assert math.isnan(row["flood_mean_lead_min"])  # no flood had a warning in force: no lead, not 0


def test_floodsense_and_the_rule_must_be_scored_over_the_same_floods():
    span = {"start": WINDOW["start"], "end": WINDOW["end"]}
    ours = floods.report_table(OURS, PUB, **span)
    theirs = floods.report_table(RULES, PUB[PUB["alert_ts"] < at("15:00")], **span)
    with pytest.raises(ValueError):
        publish.flood_rows(ours, theirs, "final_2")


def test_kept_holds_the_warned_ticks_and_the_hour_before_each_flood_only():
    # Bedok warned at 12:00, long before its floods; Tampines neither warned nor flooded
    preds = pd.concat([predictions("Bedok", {"12:00": 0.04}), predictions("Tampines", {})], ignore_index=True)
    rows = publish.kept(preds, publish.flood_hours(PUB, WINDOW["start"], WINDOW["end"]))
    hours = set(floods.hour_before(at("14:40"))) | set(floods.hour_before(at("15:05")))  # 13:40 to 15:00
    assert set(rows["ts"]) == hours | {at("12:00")}
    assert set(rows["area"]) == {"Bedok"}
    assert list(rows.columns) == ["area", "ts", "tier", "p_alert"]


def test_flood_hours_skip_floods_outside_the_window():
    keys = publish.flood_hours(PUB, DAY + pd.Timedelta(days=1), DAY + pd.Timedelta(days=2))
    assert len(keys) == 0
    assert publish.kept(OURS, keys)["tier"].ne("Low").all()


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
    assert not set(publish.DROPPED) & set(publish.TABLES)
