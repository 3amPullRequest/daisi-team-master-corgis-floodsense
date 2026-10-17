"""Gold agrees with the model contract, with silver and with the post parser, checked without Spark."""

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from utilities.bronze_sources import HEARTBEAT_TABLE
from utilities.contract_tables import (
    AREA_FEATURES_SCHEMA,
    AREA_LABELS_SCHEMA,
    FEATURE_TICKS_SCHEMA,
    LABEL_TICKS_SCHEMA,
    MODEL_ROWS_SCHEMA,
)
from utilities.forecast_categories import FORECASTS
from utilities.gold import sgt_midnight_s

from daisi_team_master_corgis_floodsense.labels import alert_text
from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.split import EMBARGO_DAYS

TRANSFORMATIONS = (
    Path(__file__).resolve().parents[1] / "src" / "pipeline" / "transformations"
)
SGT = timezone(timedelta(hours=8))


def _names(ddl: str) -> list[str]:
    return [field.split()[0] for field in re.split(r",\s*(?=\w+ [A-Z]+ COMMENT)", ddl)]


def _code(layer: str) -> str:
    return "\n".join(
        p.read_text("utf-8") for p in (TRANSFORMATIONS / layer).glob("*.py")
    )


def _resolve(name: str) -> str:
    """'C.AREA_ALERTS' -> 'area_alerts'; a quoted name as is."""
    return getattr(C, name[2:]) if name.startswith("C.") else name.strip('"')


def test_gold_schemas_are_the_contract():
    assert _names(AREA_FEATURES_SCHEMA) == list(C.AREA_FEATURES_COLUMNS)
    assert _names(LABEL_TICKS_SCHEMA) == list(C.LABEL_TICKS_COLUMNS)
    assert _names(AREA_LABELS_SCHEMA) == list(C.AREA_LABELS_COLUMNS)
    assert _names(FEATURE_TICKS_SCHEMA) == list(C.FEATURE_TICKS_COLUMNS)
    assert _names(MODEL_ROWS_SCHEMA) == list(C.MODEL_ROWS_COLUMNS)
    # the baseline's columns are all there
    assert {C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES} <= set(
        C.AREA_FEATURES_COLUMNS
    )


def test_model_rows_hold_every_model_feature_and_both_labels():
    # every gold feature column is a model feature, except those held back (lightning, 16 Oct)
    held_back = set(C.HELD_BACK_FEATURES)
    assert held_back <= set(C.AREA_FEATURES_COLUMNS)
    assert (
        set(C.MODEL_FEATURES)
        == set(C.AREA_FEATURES_COLUMNS) - {"area", "ts"} - held_back
    )
    assert {*C.MODEL_FEATURES, "alert_label", "flood_label", "split"} <= set(
        C.MODEL_ROWS_COLUMNS
    )
    assert not held_back & set(C.MODEL_ROWS_COLUMNS)
    # the forecast's rain classes are the ones silver assigns
    assert {rain for _, rain in FORECASTS.values()} == set(C.RAIN_CLASSES)
    assert set(C.FEATURE_MISSING_REASONS) == {C.RAIN_GAP, C.FORECAST_GAP}


def test_required_and_optional_features_split_the_model_features():
    assert set(C.REQUIRED_FEATURES) | set(C.OPTIONAL_FEATURES) == set(C.MODEL_FEATURES)
    assert not set(C.REQUIRED_FEATURES) & set(C.OPTIONAL_FEATURES)
    # the rule reads only required columns, and gold.feature_ticks covers exactly those
    assert {C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES} <= set(
        C.REQUIRED_FEATURES
    )
    # the nearby rain reads a window gold keeps per gauge, from where the area's own gauges end
    assert C.NEARBY_RAIN_MINUTES in C.RAIN_WINDOWS_MINUTES
    assert C.NEARBY_RAIN_KM[0] == C.GAUGE_RADIUS_KM < C.NEARBY_RAIN_KM[1]
    # windows that start on a tick, and lightning records that tile them
    assert C.WEATHER_MINUTES % C.TICK_MINUTES == 0
    assert C.LIGHTNING_MINUTES % C.TICK_MINUTES == 0
    assert (
        C.LIGHTNING_RECORDS_PER_WINDOW * C.LIGHTNING_RECORD_MINUTES
        == C.LIGHTNING_MINUTES
    )
    # a source that starts after the labels can only be an optional feature, never a required one
    assert C.LIGHTNING_START > C.LABELS_START


def test_split_and_label_window_constants_hold_together():
    assert C.LABELS_START <= C.TRAIN_START < C.TEST_START < C.SPLIT_END
    assert timedelta(days=EMBARGO_DAYS) >= timedelta(minutes=C.HORIZON_MINUTES)
    # a label window reaches back as far as the parser lets a stamp run ahead of its post
    assert timedelta(minutes=C.STAMP_AHEAD_MINUTES) == alert_text.CLOCK_SKEW
    # the longest stamp-to-post delay in 2021-2026 was 35.3 minutes
    assert C.POST_DELAY_MAX_MINUTES > 35.3
    assert C.READING_LAG_MINUTES % C.TICK_MINUTES == 0  # so readings count from a tick
    assert set(C.MISSING_REASONS) == {C.POSTS_NOT_RECOVERED, C.UNPLACED_ALERT}


def test_sgt_midnights_in_epoch_seconds():
    for day in (C.LABELS_START, C.TEST_START, date(2026, 1, 1)):
        midnight = datetime(day.year, day.month, day.day, tzinfo=SGT)
        assert sgt_midnight_s(day) == int(midnight.timestamp())


def test_gold_reads_only_tables_that_exist():
    gold, silver = _code("gold"), _code("silver")
    silver_defined = set(re.findall(r'name=f"\{SILVER\}\.([a-z_0-9]+)"', silver)) | set(
        re.findall(r'TARGET = f"\{SILVER\}\.([a-z_0-9]+)"', silver)
    )
    silver_read = {
        _resolve(n)
        for n in re.findall(r'silver\(spark, ("[a-z_0-9]+"|C\.[A-Z_]+)\)', gold)
    }
    assert silver_read and silver_read <= silver_defined, silver_read - silver_defined
    bronze_read = {
        _resolve(n) for n in re.findall(r'bronze\(spark, ("[a-z_0-9]+")\)', gold)
    }
    assert bronze_read == {HEARTBEAT_TABLE}
    gold_defined = {
        _resolve(n) for n in re.findall(r'name=("[a-z_0-9]+"|C\.[A-Z_]+)', gold)
    }
    gold_read = {
        _resolve(n)
        for n in re.findall(r'spark\.read\.table\(("[a-z_0-9]+"|C\.[A-Z_]+)\)', gold)
    }
    assert gold_read <= gold_defined, gold_read - gold_defined
    assert {
        C.AREA_FEATURES,
        C.LABEL_TICKS,
        C.AREA_LABELS,
        C.FEATURE_TICKS,
        C.MODEL_ROWS,
    } <= gold_defined
