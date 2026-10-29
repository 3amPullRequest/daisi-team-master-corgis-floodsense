"""gold.area_features: what each NEA forecast area had seen at each 5-minute tick (contract.AREA_FEATURES_COLUMNS): the
rain near it, NEA's 2-hour forecast for it, the rain 5-20 km away and upwind of it, the island-wide weather, the
lightning near it (gold.tick_weather, gold.area_lightning in weather.py) and the predicted tide (gold.tide_levels). Also
gold.area_rain_stations (the gauges within reach of each area), gold.area_forecasts (the forecast valid at each tick)
and gold.feature_ticks (whether a tick's required features are known).

A reading counts at the ticks whose windows hold it, from contract.READING_LAG_MINUTES after its timestamp, and not
before NEA published it where NEA says when (silver.rainfall_5min.published_at, the CSV years). A forecast counts from
the tick at or after its publication (updated_at) until a newer one is published or its valid period ends. So a tick
sees only what a live fetch at that moment could have had, and replays of past storms are honest. Sums are exact
decimals and the max over gauges doesn't depend on order, so a rebuild gives the same values however Spark splits the
work.

The definitions live in the floodsense package (features.py), which flood-live's score task shares, so a live
prediction reads exactly what the models learnt from. The tables, in the order they depend on each other:

- area_rain_stations: which gauges lie within GAUGE_RADIUS_KM of each area's label point, with their distance. An area
  with no gauge in reach keeps a row of nulls, which an expectation counts.
- gauge_rain: per gauge and tick, the rain total over each window (15, 30, 60 minutes). Rather than joining readings to
  time ranges, each reading is expanded into the ticks whose windows hold it: it first counts at the tick
  READING_LAG_MINUTES after its timestamp and at each of the next ticks up to the longest window, as slot 0, 1, 2 ...
  A reading is dropped from the ticks before NEA published it. A window's total is then a sum over the slots it spans,
  a plain group-by, and it cannot contain a reading its live twin hadn't yet received. Beside them, the rain of the
  hours before the last one (PRIOR_RAIN_HOURS, 23 Oct): the gauge's hour totals at the ticks 1, 2, ... hours before,
  summed by a window over the ticks that share the tick's second of the hour, and null where an hour has no row. And
  the rain still to drain at each of DRAIN_RATES_MM_H (24 Oct): a leaky bucket that each reading fills at the first
  tick it counts and that empties at the rate, by two running windows per gauge (features.rain_surplus). And the
  drains' response (25 Oct): the flow into the drains and its overflow, from a nonlinear reservoir per gauge that the
  same readings fill, run in pandas a gauge at a time (features.drain_response).
- area_forecasts: each silver forecast covers the ticks from its publication until the next one for the area (or the
  end of its valid period), which gives the forecast valid at every tick and its age.
- area_nearby_rain: for the gauges between GAUGE_RADIUS_KM and NEARBY_RAIN_KM[1] away, the largest 30-minute total, and
  the largest among those upwind. A gauge is upwind when its bearing from the area is within UPWIND_HALF_ANGLE_DEG of
  the direction the island-wide wind blows from, and the wind is at least UPWIND_MIN_WIND_KN.
- area_nearby_alerts (23 Oct): how many PUB alerts were posted for other areas at places within NEARBY_ALERT_WINDOWS'
  distances of each area's label point, in each window ending POST_LAG_MINUTES before the tick. Like a reading, each
  alert is expanded into the ticks whose windows hold it.
- area_features: per area and tick, the largest total among the gauges in reach for each window (the max, not the mean,
  because a flash flood follows the most intense cell), for the hours before the last one (among the gauges whose
  hours are all there), for the rain still to drain and for the drains' flows and overflows, how long the storm over
  the area has gone on and how far each flow is above its highest in it (windows over the area's ticks), joined to the
  forecast (its class, and the class in fewer groups), the nearby and upwind rain, PUB's
  alerts nearby, the island-wide weather (the dew point among it), the lightning, the tide and the area's ground from
  the national map (gold.area_terrain in terrain.py, one row per area). A row exists where a
  gauge in reach had a reading in the last hour. The optional columns are null where their source had nothing: no lightning before its feed began or when it was
  down for part of the window, no upwind rain without a wind, no count of PUB's alerts nearby where gold.label_ticks
  says a post of the window may be lost (an hour before the tick isn't recorded).
- feature_ticks: for every tick of gold.label_ticks, whether its features are known. An outage leaves no rows rather
  than zero rain, so a tick is `missing` unless the gauge network reported at every reading timestamp its rain windows
  read (else RAIN_GAP) and a forecast was valid (else FORECAST_GAP).
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.contract_tables import AREA_FEATURES_SCHEMA, FEATURE_TICKS_SCHEMA
from utilities.gold import TICK_S, seconds, silver
from utilities.silver import distance_km

from daisi_team_master_corgis_floodsense import features as F_
from daisi_team_master_corgis_floodsense.features import LAG_S, SLOTS
from daisi_team_master_corgis_floodsense.ml import contract as C


@dp.materialized_view(
    name="area_rain_stations",
    comment=f"NEA rain gauges within {C.GAUGE_RADIUS_KM:g} km of each forecast area's label point: the gauges whose "
    "rain gold.area_features reads for the area, with the distance and the first and last day each reported.",
)
@dp.expect("every_area_reaches_a_gauge", "station_id IS NOT NULL")
def area_rain_stations():
    areas = spark.read.table(silver(spark, "forecast_areas")).select(
        "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
    )
    gauges = spark.read.table(silver(spark, "rain_stations")).select(
        "station_id",
        F.col("name").alias("station_name"),
        "lat",
        "lon",
        "first_day",
        "last_day",
    )
    km = distance_km(F.col("area_lat"), F.col("area_lon"), F.col("lat"), F.col("lon"))
    within = (
        areas.crossJoin(gauges)  # 47 areas x ~100 gauges
        .withColumn("km", F.round(km, 3))
        .where(F.col("km") <= C.GAUGE_RADIUS_KM)
        .drop("area_lat", "area_lon", "lat", "lon")
    )
    # an area with no gauge in reach keeps one row of nulls, which the expectation counts
    return (
        areas.select("area")
        .join(within, "area", "left")
        .select("area", "station_id", "station_name", "km", "first_day", "last_day")
    )


@dp.materialized_view(
    name="area_forecasts",
    comment="NEA's 2-hour forecast for each area at each 5-minute tick, as a live fetch at the tick would have had "
    "it: the newest forecast published (updated_at) by the tick, while its valid period lasts. One row per area and "
    "tick with a valid forecast. gold.area_features and gold.feature_ticks read it.",
    cluster_by=["ts", "area"],
)
@dp.expect("rain_class_known", "forecast_rain_class BETWEEN 0 AND 5")
def area_forecasts():
    return F_.forecast_ticks(spark.read.table(silver(spark, "forecasts_2h")))


@dp.materialized_view(name="gauge_rain", private=True)
def gauge_rain():
    """Per gauge and tick, its rain totals over each of RAIN_WINDOWS_MINUTES (w15, w30, w60) and over the hours before
    the last one (p3, p6, p24), the rain still to drain at each of DRAIN_RATES_MM_H (s10 ... s100), and the drains'
    flows and overflows (d10, d30, o10, o30), from the readings NEA had published by each tick (features.gauge_rain).
    gold.area_features reads the gauges within reach of each area, area_nearby_rain those beyond."""
    return F_.gauge_rain(spark.read.table(silver(spark, "rainfall_5min")))


@dp.materialized_view(name="area_nearby_rain", private=True)
def area_nearby_rain():
    """Per area and tick, the largest NEARBY_RAIN_MINUTES total among the gauges NEARBY_RAIN_KM away, and among those
    of them upwind (features.nearby_rain). Rows only where one of them rained; gold.area_features reads a missing row
    as none."""
    return F_.nearby_rain(
        spark.read.table("gauge_rain"),
        spark.read.table(silver(spark, "forecast_areas")),
        spark.read.table(silver(spark, "rain_stations")),
        spark.read.table("tick_weather"),
    )


@dp.materialized_view(name="area_nearby_alerts", private=True)
def area_nearby_alerts():
    """Per area and tick, how many PUB alerts were posted for other areas at places near it in each of
    contract.NEARBY_ALERT_WINDOWS (features.nearby_alerts). Rows only where one counted; gold.area_features reads a
    missing row as none, where gold.label_ticks says no post can be missing."""
    return F_.nearby_alerts(
        spark.read.table(silver(spark, C.AREA_ALERTS)),
        spark.read.table(silver(spark, "forecast_areas")),
    )


@dp.materialized_view(
    name=C.AREA_FEATURES,
    comment="What each NEA forecast area had seen at each 5-minute tick: the largest 15/30/60-minute rain total among "
    f"the gauges within {C.GAUGE_RADIUS_KM:g} km, from the readings published by the tick (READING_LAG_MINUTES = "
    f"{C.READING_LAG_MINUTES}), and NEA's 2-hour forecast valid at the tick; then the rain of the 2, 5 and 23 hours "
    "before the last one, how long the storm over the area has gone on, the rain still to drain at "
    f"{', '.join(map(str, C.DRAIN_RATES_MM_H))} mm/h, the drains' flow and overflow at responses of "
    f"{' and '.join(map(str, C.DRAIN_RESPONSE_MINUTES))} min and each flow's rise in the storm, the forecast's class "
    "in fewer groups, the rain "
    f"{C.NEARBY_RAIN_KM[0]:g}-{C.NEARBY_RAIN_KM[1]:g} km away and upwind, PUB's "
    "alerts just posted for nearby areas, the island-wide weather (with the dew point), and the lightning near it, "
    "which are null where their source has no "
    "data; the predicted tide at Tanjong Pagar with the minutes to its next high water; and the area's ground from "
    "the national map (gold.area_terrain), the same at every tick. One row per area and "
    "tick wherever a gauge in reach had a reading in the last hour; a missing row means no data, not a dry tick "
    "(gold.feature_ticks). What the rainfall rule, the gated LightGBM pair and the ordinal model read.",
    schema=AREA_FEATURES_SCHEMA,
    cluster_by=["ts", "area"],
)
@dp.expect_all(
    {
        "rain_nonnegative": "rain_60min_mm >= 0",
        "windows_nested": "coalesce(rain_15min_mm, 0) <= coalesce(rain_30min_mm, 0) "
        "AND coalesce(rain_30min_mm, 0) <= rain_60min_mm",
        "rain_plausible": "rain_60min_mm < 250",
        # the prior rain is null where no gauge reported in every hour; where known it is a sum of totals
        **{
            f"{c}_nonnegative": f"{c} IS NULL OR {c} >= 0"
            for c in C.PRIOR_RAIN
        },
        "prior_day_plausible": f"{C.PRIOR_RAIN[-1]} IS NULL OR {C.PRIOR_RAIN[-1]} < 1000",
        # a storm's age runs from 0 to its cap; a slower drain leaves more to drain, never less
        "storm_age_in_range": f"{C.STORM_AGE} IS NULL OR {C.STORM_AGE} BETWEEN 0 AND {C.STORM_MAX_MINUTES}",
        "surplus_nested": f"{C.RAIN_SURPLUS[-1]} >= 0 AND "
        + " AND ".join(f"{slow} >= {fast}" for slow, fast in zip(C.RAIN_SURPLUS, C.RAIN_SURPLUS[1:])),
        # the drains' flows and overflows are known wherever the rain is (gauge_rain's pandas step gave every row
        # one), never below 0; a rise only in a storm, never above its flow
        "drains_known": " AND ".join(f"{c} IS NOT NULL AND {c} >= 0" for c in C.DRAIN_STATE),
        "rise_in_storm": " AND ".join(
            f"({rise} IS NULL) = ({C.STORM_AGE} IS NULL) AND coalesce({rise} <= {flow}, TRUE)"
            for flow, rise in zip(C.DRAIN_FLOW, C.DRAIN_RISE)
        ),
        **{
            f"{name}_known": f"({name} IS NULL) = (forecast_rain_class IS NULL)"
            for name in C.FORECAST_GROUPED
        },
        "forecast_fresh": "forecast_age_min IS NULL OR forecast_age_min BETWEEN 0 AND 120",
        "upwind_is_nearby": "rain_upwind_30min_mm IS NULL OR rain_upwind_30min_mm <= rain_nearby_30min_mm",
        # the nearby-alert windows nest: 5 km in 30 min within 5 km in 60 within 10 km in 60
        "pub_alerts_nested": f"{C.PUB_ALERTS[0]} IS NULL OR ({C.PUB_ALERTS[0]} >= 0 AND "
        f"{C.PUB_ALERTS[0]} <= {C.PUB_ALERTS[1]} AND {C.PUB_ALERTS[1]} <= {C.PUB_ALERTS[2]})",
        "lightning_nested": f"{F_.LIGHTNING[0]} IS NULL OR {F_.LIGHTNING[0]} <= {F_.LIGHTNING[-1]}",
        "tide_known": "tide_height_m IS NOT NULL AND minutes_to_high_tide IS NOT NULL",
        "high_tide_within_a_day": "minutes_to_high_tide BETWEEN 0 AND 1560",  # 26 h: some days have one high water
        # every area has land on the national map, so its ground is known at every tick
        "ground_known": " AND ".join(f"{c} IS NOT NULL" for c in C.MAP_FEATURES),
    }
)
def area_features():
    return F_.area_features(
        spark.read.table("gauge_rain"),
        spark.read.table("area_rain_stations"),
        spark.read.table("area_forecasts"),
        spark.read.table("tide_levels"),
        spark.read.table("area_nearby_rain"),
        spark.read.table("tick_weather"),
        spark.read.table("area_lightning"),
        alerts_nearby=spark.read.table("area_nearby_alerts"),
        posts_recorded=F_.recorded_posts(spark.read.table(C.LABEL_TICKS)),
        terrain=spark.read.table(C.AREA_TERRAIN),
    )


@dp.materialized_view(
    name=C.FEATURE_TICKS,
    comment="One row per tick of gold.label_ticks: whether the tick's features are known (contract.FEATURE_TICKS_COLUMNS). "
    "'missing' where the gauge network didn't report at every reading timestamp the tick's rain windows read, or no "
    "2-hour forecast was valid: NEA outages, and days the API backfill hasn't landed yet. The same for every area.",
    schema=FEATURE_TICKS_SCHEMA,
    cluster_by=["ts"],
)
@dp.expect(
    "known_reason",
    f"missing_reason IS NULL OR missing_reason IN {C.FEATURE_MISSING_REASONS}",
)
def feature_ticks():
    grid = spark.read.table(C.LABEL_TICKS).select(seconds("ts").alias("tick_s"))
    # a reading timestamp at which any gauge reported counts for the ticks whose windows read it, published or not:
    # this is about the feed being up, not about what had arrived by the tick
    stamps = (
        spark.read.table(silver(spark, "rainfall_5min"))
        .where("rain_mm IS NOT NULL AND ts IS NOT NULL AND station_id IS NOT NULL")
        .select(seconds("ts").alias("reading_s"))
        .distinct()
    )
    rain = (
        stamps.select(
            "reading_s", F.explode(F.sequence(F.lit(0), F.lit(SLOTS - 1))).alias("slot")
        )
        .groupBy((F.col("reading_s") + LAG_S + F.col("slot") * TICK_S).alias("tick_s"))
        .agg(F.count("*").alias("rain_stamps"))
    )
    forecasts = (
        spark.read.table("area_forecasts")
        .groupBy(seconds("ts").alias("tick_s"))
        .agg(F.count("*").alias("forecast_areas"))
    )
    rows = grid.join(rain, "tick_s", "left").join(forecasts, "tick_s", "left")
    rain_stamps = F.coalesce(F.col("rain_stamps"), F.lit(0))
    forecast_areas = F.coalesce(F.col("forecast_areas"), F.lit(0))
    reason = F.when(rain_stamps < SLOTS, C.RAIN_GAP).when(
        forecast_areas == 0, C.FORECAST_GAP
    )
    return rows.select(
        F.timestamp_seconds("tick_s").alias("ts"),
        rain_stamps.cast("int").alias("rain_stamps"),
        forecast_areas.cast("int").alias("forecast_areas"),
        F.when(reason.isNull(), C.RECORDED)
        .otherwise(C.MISSING)
        .alias("feature_window"),
        reason.alias("missing_reason"),
    )
