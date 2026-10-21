"""gold.area_features: what each NEA forecast area had seen at each 5-minute tick (contract.AREA_FEATURES_COLUMNS): the
rain near it, NEA's 2-hour forecast for it and the predicted tide (gold.tide_levels). Also gold.area_rain_stations (the
gauges within reach of each area), gold.area_forecasts (the forecast valid at each tick) and gold.feature_ticks
(whether a tick's features are known).

A reading counts at the ticks whose windows hold it, from contract.READING_LAG_MINUTES after its timestamp, and not
before NEA published it where NEA says when (silver.rainfall_5min.published_at, the CSV years). A forecast counts from
the tick at or after its publication (updated_at) until a newer one is published or its valid period ends. So a tick
sees only what a live fetch at that moment could have had, and replays of past storms are honest. Sums are exact
decimals and the max over gauges doesn't depend on order, so a rebuild gives the same values however Spark splits the
work.

The definitions live in the floodsense package (features.py), which flood-live's score task shares, so a live
prediction reads exactly what the models learnt from.
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


@dp.materialized_view(
    name=C.AREA_FEATURES,
    comment="What each NEA forecast area had seen at each 5-minute tick: the largest 15/30/60-minute rain total among "
    f"the gauges within {C.GAUGE_RADIUS_KM:g} km, from the readings published by the tick (READING_LAG_MINUTES = "
    f"{C.READING_LAG_MINUTES}), NEA's 2-hour forecast valid at the tick, and the predicted tide at Tanjong Pagar "
    "with the minutes to its next high water. One row per area and tick wherever a "
    "gauge in reach had a reading in the last hour; a missing row means no data, not a dry tick (gold.feature_ticks). "
    "What the rainfall rule and the gated LightGBM pair read.",
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
        "tide_known": "tide_height_m IS NOT NULL AND minutes_to_high_tide IS NOT NULL",
        "high_tide_within_a_day": "minutes_to_high_tide BETWEEN 0 AND 1560",  # 26 h: some days have one high water
    }
)
def area_features():
    return F_.area_features(
        spark.read.table(silver(spark, "rainfall_5min")),
        spark.read.table("area_rain_stations"),
        spark.read.table("area_forecasts"),
        spark.read.table("tide_levels"),
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
