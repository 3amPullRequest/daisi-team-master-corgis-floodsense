"""Gold agrees with the model contract, with silver and with the post parser, checked without Spark."""

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
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


def test_the_prior_rain_and_dew_point_are_optional_model_features():
    # 23 Oct: the rain of the hours before the last one, and the dew point in place of relative humidity
    assert C.PRIOR_RAIN == ("rain_prior_3h_mm", "rain_prior_6h_mm", "rain_prior_24h_mm")
    assert {*C.PRIOR_RAIN, "dew_point_c"} <= set(C.OPTIONAL_FEATURES)
    # the base set is what v5 reads; the extended one swaps the humidity for the dew point and adds the prior rain;
    # PUB's alerts nearby, the storm so far and the forecast groups are in neither, read only where a spec adds them
    assert set(C.BASE_FEATURES) | set(C.EXTENDED_FEATURES) | set(C.OUTSIDE_SETS) == set(C.MODEL_FEATURES)
    assert not {*C.PRIOR_RAIN, "dew_point_c"} & set(C.BASE_FEATURES) and "rel_humidity_pct" in C.BASE_FEATURES
    assert {*C.PRIOR_RAIN, "dew_point_c"} <= set(C.EXTENDED_FEATURES) and "rel_humidity_pct" not in C.EXTENDED_FEATURES
    # both keep gold's order, so a pair saved on either still loads (GatedPair.load)
    for features in (C.BASE_FEATURES, C.EXTENDED_FEATURES):
        assert list(features) == [c for c in C.MODEL_FEATURES if c in features]
    # the Magnus dew point: equal to the air at saturation, about 24 deg C at 30 deg C and 70%
    assert C.dew_point_c(25.0, 100.0) == pytest.approx(25.0)
    assert C.dew_point_c(30.0, 70.0) == pytest.approx(23.9, abs=0.1)


def test_pub_alerts_nearby_are_optional_counts_outside_both_feature_sets():
    # 23 Oct: PUB alerts posted for other areas nearby, 5 km in 30 and 60 min, 10 km in 60
    assert C.PUB_ALERTS == ("pub_alerts_5km_30min", "pub_alerts_5km_60min", "pub_alerts_10km_60min")
    assert set(C.PUB_ALERTS) <= set(C.OPTIONAL_FEATURES)
    assert not set(C.PUB_ALERTS) & (set(C.BASE_FEATURES) | set(C.EXTENDED_FEATURES))
    assert C.FEATURE_GROUPS["pub_alerts"] == C.PUB_ALERTS
    # gold's area_alerts carry the place, which the counts need
    assert {"lat", "lon"} <= set(C.AREA_ALERTS_COLUMNS)


def test_the_storm_so_far_and_the_forecast_groups_are_optional_and_in_neither_set():
    # 24 Oct: how long the storm has gone on, the rain still to drain at four rates, and the forecast in fewer groups
    assert C.STORM_HISTORY == (
        "storm_age_min",
        "rain_surplus_10mmh_mm",
        "rain_surplus_25mmh_mm",
        "rain_surplus_50mmh_mm",
        "rain_surplus_100mmh_mm",
    )
    assert C.FORECAST_GROUPED == ("forecast_rain_group3", "forecast_rain_group4")
    new = {*C.STORM_HISTORY, *C.FORECAST_GROUPED}
    assert new <= set(C.OPTIONAL_FEATURES)
    assert not new & (set(C.BASE_FEATURES) | set(C.EXTENDED_FEATURES))
    assert C.FEATURE_GROUPS["storm"] == (C.STORM_AGE,)
    assert C.FEATURE_GROUPS["surplus"] == C.RAIN_SURPLUS
    assert C.FEATURE_GROUPS["forecast_groups"] == C.FORECAST_GROUPED
    # each grouping gives every rain class a group, from 0 up in steps of at most one, never down: the groups keep the
    # classes' order, so the chance may only rise with them as it does with the class
    for name, grouping in C.FORECAST_GROUPINGS.items():
        assert len(grouping) == len(C.RAIN_CLASSES) and grouping[0] == 0
        assert all(0 <= b - a <= 1 for a, b in zip(grouping, grouping[1:]))
        assert name.endswith(str(len(set(grouping))))
    # gold's surplus_nested expectation reads the rates slowest first: a slower drain leaves more to drain
    assert list(C.DRAIN_RATES_MM_H) == sorted(set(C.DRAIN_RATES_MM_H))
    # the storm reads a window gold keeps, and its capped age fits in the day of readings live scoring reads
    assert C.STORM_WINDOW_MINUTES in C.RAIN_WINDOWS_MINUTES
    assert C.STORM_MAX_MINUTES + C.STORM_GAP_MINUTES <= (max(C.PRIOR_RAIN_HOURS) - 1) * 60


def test_the_drains_response_is_optional_and_in_neither_set():
    # 25 Oct: per gauge, a nonlinear reservoir's flow into the drains and its overflow beyond capacity, and how far each
    # flow is above its highest earlier in the storm
    assert C.DRAIN_RESPONSE == (
        "drain_flow_10min_mmh",
        "drain_flow_30min_mmh",
        "drain_overflow_10min_mm",
        "drain_overflow_30min_mm",
        "drain_rise_10min_mmh",
        "drain_rise_30min_mmh",
    )
    assert set(C.DRAIN_RESPONSE) <= set(C.OPTIONAL_FEATURES)
    assert not set(C.DRAIN_RESPONSE) & (set(C.BASE_FEATURES) | set(C.EXTENDED_FEATURES))
    assert C.FEATURE_GROUPS["drain"] == C.DRAIN_STATE and C.FEATURE_GROUPS["rise"] == C.DRAIN_RISE
    # the capacity is a whole number of flow steps, so the overflow counts whole steps
    assert float(C.DRAIN_CAPACITY_MM_H / C.DRAIN_FLOW_STEP_MM_H).is_integer()


def test_the_ground_is_optional_last_and_in_neither_set():
    # 25 Oct: the national map's lines (contours and roads) and polygons (coast, waterways, parks), one value per area
    assert C.MAP_FEATURES == (*C.MAP_LINE, *C.MAP_POLYGON)
    assert set(C.MAP_FEATURES) <= set(C.OPTIONAL_FEATURES)
    assert not set(C.MAP_FEATURES) & (set(C.BASE_FEATURES) | set(C.EXTENDED_FEATURES))
    assert C.FEATURE_GROUPS["map_line"] == C.MAP_LINE
    assert C.FEATURE_GROUPS["map_polygon"] == C.MAP_POLYGON
    # last in gold's order, so a pair saved before them reads its inputs in the same order (GatedPair.load)
    assert C.MODEL_FEATURES[-len(C.MAP_FEATURES) :] == C.MAP_FEATURES
    # gold.area_terrain makes every column under its contract name, and area_features reads the table
    terrain = (TRANSFORMATIONS / "gold" / "terrain.py").read_text("utf-8")
    assert all(f'"{c}"' in terrain for c in C.MAP_FEATURES)
    assert "terrain=spark.read.table(C.AREA_TERRAIN)" in _code("gold")


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
    assert set(C.MISSING_REASONS) == {
        C.POSTS_NOT_RECOVERED,
        C.UNREAD_POST,
        C.UNPLACED_ALERT,
    }


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
