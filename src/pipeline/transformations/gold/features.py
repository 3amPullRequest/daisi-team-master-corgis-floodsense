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
  a plain group-by, and it cannot contain a reading its live twin hadn't yet received.
- area_forecasts: each silver forecast covers the ticks from its publication until the next one for the area (or the
  end of its valid period), which gives the forecast valid at every tick and its age.
- area_nearby_rain: for the gauges between GAUGE_RADIUS_KM and NEARBY_RAIN_KM[1] away, the largest 30-minute total, and
  the largest among those upwind. A gauge is upwind when its bearing from the area is within UPWIND_HALF_ANGLE_DEG of
  the direction the island-wide wind blows from, and the wind is at least UPWIND_MIN_WIND_KN.
- area_features: per area and tick, the largest total among the gauges in reach for each window (the max, not the mean,
  because a flash flood follows the most intense cell), joined to the forecast, the nearby and upwind rain, the
  island-wide weather, the lightning and the tide. A row exists where a gauge in reach had a reading in the last hour.
  The optional columns are null where their source had nothing: no lightning before its feed began or when it was down
  for part of the window, no upwind rain without a wind.
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
    """Per gauge and tick, its rain totals over each of RAIN_WINDOWS_MINUTES (w15, w30, w60), from the readings NEA had
    published by the tick (features.gauge_rain). gold.area_features reads the gauges within reach of each area,
    area_nearby_rain those beyond."""
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


@dp.materialized_view(
    name=C.AREA_FEATURES,
    comment="What each NEA forecast area had seen at each 5-minute tick: the largest 15/30/60-minute rain total among "
    f"the gauges within {C.GAUGE_RADIUS_KM:g} km, from the readings published by the tick (READING_LAG_MINUTES = "
    f"{C.READING_LAG_MINUTES}), and NEA's 2-hour forecast valid at the tick; then the rain "
    f"{C.NEARBY_RAIN_KM[0]:g}-{C.NEARBY_RAIN_KM[1]:g} km away and upwind, the island-wide weather, and the lightning "
    "near it, which are null where their source has no data; and the predicted tide at Tanjong Pagar with the minutes "
    "to its next high water. One row per area and tick wherever a gauge in reach had a reading in the last hour; a "
    "missing row means no data, not a dry tick (gold.feature_ticks). What the rainfall rule and the gated LightGBM "
    "pair read.",
    schema=AREA_FEATURES_SCHEMA,
    cluster_by=["ts", "area"],
)
@dp.expect_all(
    {
        "rain_nonnegative": "rain_60min_mm >= 0",
        "windows_nested": "coalesce(rain_15min_mm, 0) <= coalesce(rain_30min_mm, 0) "
        "AND coalesce(rain_30min_mm, 0) <= rain_60min_mm",
        "rain_plausible": "rain_60min_mm < 250",
        "forecast_fresh": "forecast_age_min IS NULL OR forecast_age_min BETWEEN 0 AND 120",
        "upwind_is_nearby": "rain_upwind_30min_mm IS NULL OR rain_upwind_30min_mm <= rain_nearby_30min_mm",
        "lightning_nested": f"{F_.LIGHTNING[0]} IS NULL OR {F_.LIGHTNING[0]} <= {F_.LIGHTNING[-1]}",
        "tide_known": "tide_height_m IS NOT NULL AND minutes_to_high_tide IS NOT NULL",
        "high_tide_within_a_day": "minutes_to_high_tide BETWEEN 0 AND 1560",  # 26 h: some days have one high water
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
