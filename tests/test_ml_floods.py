"""floods.flood_table: per flash flood, the model's lead time and its tier and chance before the flood."""

import math

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import floods

DAY = pd.Timestamp("2025-11-20")
WINDOW = {"start": DAY, "end": DAY + pd.Timedelta(days=1)}


def at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY.date()} {hhmm}")


def predictions(area: str, p_by_tick: dict[str, float], medium: float = 0.05, high: float = 0.5):
    """Every tick of the day for `area`, at p 0.001 (Low) unless `p_by_tick` says otherwise."""
    ticks = pd.date_range(DAY, DAY + pd.Timedelta(days=1), freq="5min", inclusive="left")
    p = pd.Series(0.001, index=ticks)
    for hhmm, value in p_by_tick.items():
        p[at(hhmm)] = value
    tier = np.select([p >= high, p >= medium], ["High", "Medium"], "Low")
    return pd.DataFrame({"area": area, "ts": ticks, "tier": tier, "p_alert": p.to_numpy()})


def alerts(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(area, at(hhmm), kind) for area, hhmm, kind in rows], columns=["area", "alert_ts", "kind"]
    )


def test_lead_time_runs_from_the_first_warning_in_force_to_the_flood():
    # Medium from 14:00, PUB's warning at 14:25, the flash flood at 14:40
    preds = predictions("Bedok", {f"14:{m:02d}": 0.08 for m in range(0, 40, 5)})
    pub = alerts(("Bedok", "14:25", C.WARNING), ("Bedok", "14:40", C.FLASH_FLOOD))
    row = floods.flood_table(preds, pub, **WINDOW).iloc[0]
    assert row["flood_ts"] == at("14:40") and row["first_alert"] == at("14:25")
    assert row["warned_from"] == at("14:00")
    assert row["lead_min"] == 40  # minutes from the first Medium tick to the flood
    assert row["pub_lead_min"] == 15  # PUB's own warning came 15 minutes before the flood
    assert row["before_pub"]  # the model warned 25 minutes before PUB did
    assert row["public_warning_min"] == 40  # the public's warning: whichever came first
    assert row["judged"]


def test_a_warning_still_counts_for_an_hour_after_its_last_tick():
    # one Medium tick at 13:45: in force until 14:45, so a flood at 14:40 was warned 55 minutes ahead
    preds = predictions("Bedok", {"13:45": 0.08})
    row = floods.flood_table(preds, alerts(("Bedok", "14:40", C.FLASH_FLOOD)), **WINDOW).iloc[0]
    assert row["lead_min"] == 55 and row["before_pub"]


def test_a_flood_with_no_warning_in_force_is_a_miss_without_lead():
    # the last Medium tick at 13:00 lapsed at 14:00, forty minutes before the flood
    preds = predictions("Bedok", {"13:00": 0.08, "14:35": 0.02})
    row = floods.flood_table(preds, alerts(("Bedok", "14:40", C.FLASH_FLOOD)), **WINDOW).iloc[0]
    assert pd.isna(row["warned_from"]) and math.isnan(row["lead_min"])
    assert not row["before_pub"]
    assert row["tier_at_flood"] == "Low" and row["p_alert_at_flood"] == pytest.approx(0.02)


def test_each_flood_shows_the_tier_and_chance_at_its_last_tick_and_their_peak_in_the_hour():
    # a flood at 14:33: its last tick is 14:30, and its hour runs over the 12 ticks 13:35 to 14:30
    preds = predictions("Bedok", {"13:30": 0.9, "13:35": 0.06, "14:05": 0.6, "14:30": 0.07})
    table = floods.flood_table(preds, alerts(("Bedok", "14:33", C.FLASH_FLOOD)), **WINDOW)
    row = table.iloc[0]
    assert list(floods.hour_before(at("14:33"))) == list(pd.date_range(at("13:35"), at("14:30"), freq="5min"))
    assert row["tier_at_flood"] == "Medium" and row["p_alert_at_flood"] == pytest.approx(0.07)
    assert row["peak_tier"] == "High" and row["peak_p_alert"] == pytest.approx(0.6)  # 13:30's 0.9 is outside it
    assert row["ticks_scored"] == 12
    assert list(table.columns) == floods.COLUMNS


def test_every_flood_gets_a_row_and_a_flood_the_model_never_scored_shows_unknown():
    preds = predictions("Bedok", {"14:30": 0.07})
    pub = alerts(
        ("Bedok", "14:40", C.FLASH_FLOOD),
        ("Tanglin", "16:00", C.WARNING),  # a warning alone is no flood: no row
        ("Tanglin", "16:20", C.FLASH_FLOOD),  # no predictions for Tanglin at all
    )
    table = floods.flood_table(preds, pub, **WINDOW).set_index("area")
    assert list(table.index) == ["Bedok", "Tanglin"]
    unscored = table.loc["Tanglin"]
    assert unscored["ticks_scored"] == 0
    assert pd.isna(unscored["tier_at_flood"]) and pd.isna(unscored["peak_p_alert"])
    assert math.isnan(unscored["lead_min"])  # no warning: a miss, as the scoreboard counts it


def test_a_flood_in_a_missing_label_window_is_listed_but_not_judged():
    preds = predictions("Bedok", {"14:30": 0.07})
    missing = pd.Series(pd.date_range(at("13:40"), at("14:40"), freq="5min"))
    table = floods.flood_table(preds, alerts(("Bedok", "14:40", C.FLASH_FLOOD)), **WINDOW, missing=missing)
    assert len(table) == 1 and not table["judged"].iloc[0]
    assert floods.summary(table)["floods_judged"] == 0


def test_summary_counts_judged_floods_and_their_lead():
    preds = pd.concat(
        [
            predictions("Bedok", {f"14:{m:02d}": 0.08 for m in range(0, 40, 5)}),
            predictions("Tanglin", {}),
        ]
    )
    pub = alerts(("Bedok", "14:40", C.FLASH_FLOOD), ("Tanglin", "16:20", C.FLASH_FLOOD))
    s = floods.summary(floods.flood_table(preds, pub, **WINDOW), prefix="oof_")
    assert s["oof_floods_judged"] == 2 and s["oof_floods_warned"] == 1
    assert s["oof_floods_warned_before_pub"] == 1
    assert s["oof_floods_mean_lead_min"] == 20  # 40 and a miss counting 0


def test_it_takes_the_scoreboards_arguments_as_train_passes_them():
    # train.py hands scoreboard.score its alerts and window as one dict, and flood_table the same dict
    preds = predictions("Bedok", {"14:30": 0.07})
    judge = {"alerts": alerts(("Bedok", "14:40", C.FLASH_FLOOD)), **WINDOW, "missing": None, "blind": None}
    assert len(floods.flood_table(preds, **judge)) == 1


def reports(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """Flash-flood reports with the place PUB named."""
    return pd.DataFrame(
        [(area, at(hhmm), C.FLASH_FLOOD, place) for area, hhmm, place in rows],
        columns=["area", "alert_ts", "kind", "location"],
    )


def test_every_flash_flood_report_counts_on_its_own_even_minutes_apart_in_one_area():
    # Tanglin warned at 16:30 and 16:35; two roads flooded at 17:11 and 17:15, after PUB's drain warning at 16:47
    preds = predictions("Tanglin", {"16:30": 0.08, "16:35": 0.08})
    pub = pd.concat(
        [
            alerts(("Tanglin", "16:47", C.WARNING)),
            reports(("Tanglin", "17:11", "King's Road"), ("Tanglin", "17:15", "Coronation Road")),
        ],
        ignore_index=True,
    )
    table = floods.report_table(preds, pub, **WINDOW)
    assert list(table["location"]) == ["King's Road", "Coronation Road"]  # a drain warning is no flood: no row
    assert list(table["warned_from"]) == [at("16:30")] * 2 and list(table["lead_min"]) == [41, 45]
    assert table["caught"].all() and table["judged"].all()
    assert list(table.columns) == floods.REPORT_COLUMNS


def test_a_flood_is_caught_only_when_the_warning_began_5_minutes_to_an_hour_before_it():
    preds = pd.concat(
        [
            predictions("Woodlands", {"15:10": 0.08}),  # 3 minutes before its flood: as it happened
            predictions("Bedok", {"14:35": 0.08}),  # 5 minutes before
            predictions("Tampines", {"13:40": 0.08}),  # an hour before: in force until 14:40, the flood's minute
            predictions("Marine Parade", {"09:45": 0.08, "09:50": 0.08}),  # in force until 10:50, begun 62 min before
            predictions("Lim Chu Kang", {}),
        ],
        ignore_index=True,
    )
    pub = reports(
        ("Woodlands", "15:13", "Riverside Road"),
        ("Bedok", "14:40", "Bedok North Avenue 3"),
        ("Tampines", "14:40", "Tampines Avenue 10"),
        ("Marine Parade", "10:47", "Mountbatten Road"),
        ("Lim Chu Kang", "15:38", "Neo Tiew Road"),
    )
    table = floods.report_table(preds, pub, **WINDOW).set_index("area")
    assert table["lead_min"].to_dict() == pytest.approx(
        {"Marine Parade": 62, "Bedok": 5, "Tampines": 60, "Woodlands": 3, "Lim Chu Kang": math.nan}, nan_ok=True
    )
    assert table["caught"].to_dict() == {
        "Marine Parade": False, "Bedok": True, "Tampines": True, "Woodlands": False, "Lim Chu Kang": False,
    }  # fmt: skip
    s = floods.report_summary(table)
    assert s["flash_floods"] == 5 and s["floods_warned"] == 4 and s["floods_caught"] == 2
    assert s["flood_mean_lead_min"] == pytest.approx((62 + 5 + 60 + 3) / 4)  # over the warned floods: no miss counts 0
    assert s["floods_caught_share"] == pytest.approx(0.4)
    assert s["floods_caught_share_lo"] < 0.4 < s["floods_caught_share_hi"]


def test_a_report_is_judged_on_what_the_model_saw_in_the_hour_before_it_not_on_lost_posts():
    preds = pd.concat([predictions("Bedok", {"14:30": 0.08}), predictions("Tanglin", {})], ignore_index=True)
    pub = reports(("Bedok", "14:40", "Bedok Road"), ("Tanglin", "17:11", "King's Road"))
    # NEA's feed was out from 13:50 to 14:00, inside Bedok's hour (13:40 to 14:40) and long before Tanglin's
    blind = pd.Series(pd.date_range(at("13:50"), at("14:00"), freq="5min"))
    table = floods.report_table(preds, pub, **WINDOW, blind=blind).set_index("area")
    assert not table.loc["Bedok", "judged"] and table.loc["Tanglin", "judged"]
    s = floods.report_summary(table)
    assert s["flash_floods"] == 1 and s["floods_warned"] == 0 and math.isnan(s["flood_mean_lead_min"])


def test_reports_outside_the_window_and_warnings_before_it_are_left_out():
    preds = predictions("Bedok", {"00:00": 0.08})
    pub = reports(("Bedok", "00:10", "Bedok Road"), ("Bedok", "23:59", "Bedok Road"))
    late = floods.report_table(preds, pub, start=at("00:05"), end=DAY + pd.Timedelta(days=1))
    assert list(late["flood_ts"]) == [at("00:10"), at("23:59")]
    assert late["warned_from"].isna().all()  # the 00:00 warning began before the window, as the scoreboard counts it
    assert floods.report_table(preds, pub.iloc[:0], **WINDOW).empty
