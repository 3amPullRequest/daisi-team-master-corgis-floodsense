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
