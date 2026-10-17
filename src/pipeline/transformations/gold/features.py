"""gold.area_features: what each NEA forecast area had seen at each 5-minute tick (contract.AREA_FEATURES_COLUMNS): the
rain near it, NEA's 2-hour forecast for it, the rain 5-20 km away and upwind of it, the island-wide weather and the
lightning near it (gold.tick_weather, gold.area_lightning in weather.py). Also gold.area_rain_stations (the gauges
within reach of each area), gold.area_forecasts (the forecast valid at each tick) and gold.feature_ticks (whether a
tick's required features are known).

A reading counts at the ticks whose windows hold it, from contract.READING_LAG_MINUTES after its timestamp, and not
before NEA published it where NEA says when (silver.rainfall_5min.published_at, the CSV years). A forecast counts from
the tick at or after its publication (updated_at) until a newer one is published or its valid period ends. So a tick
sees only what a live fetch at that moment could have had, and replays of past storms are honest. Sums are exact
decimals and the max over gauges doesn't depend on order, so a rebuild gives the same values however Spark splits the
work.

The tables, in the order they depend on each other:

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
  island-wide weather and the lightning. A row exists where a gauge in reach had a reading in the last hour. The
  optional columns are null where their source had nothing: no lightning before its feed began or when it was down
  for part of the window, no upwind rain without a wind.
- feature_ticks: for every tick of gold.label_ticks, whether its features are known. An outage leaves no rows rather
  than zero rain, so a tick is `missing` unless the gauge network reported at every reading timestamp its rain windows
  read (else RAIN_GAP) and a forecast was valid (else FORECAST_GAP).
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.contract_tables import AREA_FEATURES_SCHEMA, FEATURE_TICKS_SCHEMA
from utilities.gold import (
    TICK_S,
    seconds,
    sgt_midnight_s,
    silver,
    tick_at_or_after,
    tick_before,
    ticks_between,
)
from utilities.silver import distance_km

from daisi_team_master_corgis_floodsense.ml import contract as C

LAG_S = C.READING_LAG_MINUTES * 60
SLOTS = max(C.RAIN_WINDOWS_MINUTES) // C.TICK_MINUTES  # readings in the longest window
NEARBY = f"w{C.NEARBY_RAIN_MINUTES}"  # the gauge window the nearby and upwind rain read
LIGHTNING = tuple(C.lightning_column(km) for km in C.LIGHTNING_RADII_KM)


def _bearing_deg(lat1: str, lon1: str, lat2: str, lon2: str):
    """Initial bearing from point 1 to point 2 (columns named), degrees clockwise from north."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dl = F.radians(F.col(lon2) - F.col(lon1))
    east = F.sin(dl) * F.cos(p2)
    north = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dl)
    return F.pmod(F.degrees(F.atan2(east, north)), F.lit(360.0))


def _wind(u: str = "wind_u_kn", v: str = "wind_v_kn"):
    """(speed in knots, the direction it blows from in degrees) of a wind vector."""
    speed = F.sqrt(F.col(u) * F.col(u) + F.col(v) * F.col(v))
    blowing_from = F.pmod(F.degrees(F.atan2(F.col(u), F.col(v))) + 180, F.lit(360.0))
    return speed, blowing_from


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
    published = (
        spark.read.table(silver(spark, "forecasts_2h"))
        .where(
            "area IS NOT NULL AND updated_at IS NOT NULL AND valid_to IS NOT NULL AND rain_class IS NOT NULL"
        )
        .select(
            "area",
            "rain_class",
            seconds("updated_at").alias("published_s"),
            seconds("issued_at").alias("issued_s"),
            seconds("valid_to").alias("valid_to_s"),
        )
    )
    # the newest wins from its publication on; of two published at once, the later issue
    newer = F.lead("published_s").over(
        Window.partitionBy("area").orderBy("published_s", "issued_s")
    )
    # least() skips the null after the newest publication
    spans = published.withColumn("until_s", F.least(newer, F.col("valid_to_s")))
    ticks = spans.select(
        "area",
        "rain_class",
        "published_s",
        ticks_between(
            tick_at_or_after(F.col("published_s")), tick_before(F.col("until_s"))
        ).alias("tick_s"),
    )
    return ticks.select(
        "area",
        F.timestamp_seconds("tick_s").alias("ts"),
        F.col("rain_class").cast("int").alias("forecast_rain_class"),
        ((F.col("tick_s") - F.col("published_s")) / 60)
        .cast("double")
        .alias("forecast_age_min"),
    )


@dp.materialized_view(name="gauge_rain", private=True)
def gauge_rain():
    """Per gauge and tick, its rain totals over each of RAIN_WINDOWS_MINUTES (w15, w30, w60), from the readings NEA had
    published by the tick. gold.area_features reads the gauges within reach of each area, area_nearby_rain those
    beyond."""
    readings = spark.read.table(silver(spark, "rainfall_5min")).where(
        "rain_mm IS NOT NULL AND ts IS NOT NULL AND station_id IS NOT NULL"
    )
    # slot k: the reading is the kth newest in the windows of the tick it lands on
    counted = readings.select(
        "station_id",
        F.col("rain_mm").cast("decimal(9,2)").alias("mm"),
        seconds("published_at").alias("published_s"),
        seconds("ts").alias("reading_s"),
        F.explode(F.sequence(F.lit(0), F.lit(SLOTS - 1))).alias("slot"),
    ).withColumn("tick_s", F.col("reading_s") + LAG_S + F.col("slot") * TICK_S)
    counted = counted.where(
        F.col("published_s").isNull() | (F.col("published_s") <= F.col("tick_s"))
    )
    return counted.groupBy("station_id", "tick_s").agg(
        *(
            F.sum(F.when(F.col("slot") < w // C.TICK_MINUTES, F.col("mm"))).alias(
                f"w{w}"
            )
            for w in C.RAIN_WINDOWS_MINUTES
        )
    )


@dp.materialized_view(name="area_nearby_rain", private=True)
def area_nearby_rain():
    """Per area and tick, the largest NEARBY_RAIN_MINUTES total among the gauges NEARBY_RAIN_KM away, and among those
    of them upwind. Rows only where one of them rained; gold.area_features reads a missing row as none."""
    areas = spark.read.table(silver(spark, "forecast_areas")).select(
        "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
    )
    gauges = spark.read.table(silver(spark, "rain_stations")).select(
        "station_id", "lat", "lon"
    )
    near, far = C.NEARBY_RAIN_KM
    km = distance_km(F.col("area_lat"), F.col("area_lon"), F.col("lat"), F.col("lon"))
    pairs = (
        areas.crossJoin(gauges)  # 47 areas x ~100 gauges
        .select(
            "area",
            "station_id",
            km.alias("km"),
            _bearing_deg("area_lat", "area_lon", "lat", "lon").alias("bearing"),
        )
        .where((F.col("km") > near) & (F.col("km") <= far))
    )
    wet = (
        spark.read.table("gauge_rain")
        .where(F.col(NEARBY) > 0)
        .select("station_id", "tick_s", NEARBY)
    )
    speed, blowing_from = _wind()
    wind = (
        spark.read.table("tick_weather")
        .where("wind_u_kn IS NOT NULL")
        .select(
            seconds("ts").alias("tick_s"),
            speed.alias("wind_kn"),
            blowing_from.alias("blowing_from"),
        )
    )
    rows = wet.join(F.broadcast(pairs), "station_id").join(wind, "tick_s", "left")
    off_wind = F.abs(F.pmod(F.col("bearing") - F.col("blowing_from") + 180, 360) - 180)
    upwind = (F.col("wind_kn") >= C.UPWIND_MIN_WIND_KN) & (
        off_wind <= C.UPWIND_HALF_ANGLE_DEG
    )
    return rows.groupBy("area", "tick_s").agg(
        F.max(NEARBY).alias("nearby"),
        F.max(F.when(upwind, F.col(NEARBY))).alias("upwind"),
    )


@dp.materialized_view(
    name=C.AREA_FEATURES,
    comment="What each NEA forecast area had seen at each 5-minute tick: the largest 15/30/60-minute rain total among "
    f"the gauges within {C.GAUGE_RADIUS_KM:g} km, from the readings published by the tick (READING_LAG_MINUTES = "
    f"{C.READING_LAG_MINUTES}), and NEA's 2-hour forecast valid at the tick; then the rain "
    f"{C.NEARBY_RAIN_KM[0]:g}-{C.NEARBY_RAIN_KM[1]:g} km away and upwind, the island-wide weather, and the lightning "
    "near it, which are null where their source has no data. One row per area and tick wherever a gauge in reach had "
    "a reading in the last hour; a missing row means no data, not a dry tick (gold.feature_ticks). What the rainfall "
    "rule and the gated LightGBM pair read.",
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
        "lightning_nested": f"{LIGHTNING[0]} IS NULL OR {LIGHTNING[0]} <= {LIGHTNING[-1]}",
    }
)
def area_features():
    reach = spark.read.table("area_rain_stations").where("station_id IS NOT NULL")
    per_area = (
        spark.read.table("gauge_rain")
        .join(F.broadcast(reach.select("area", "station_id")), "station_id")
        .groupBy("area", "tick_s")
        .agg(
            *(
                F.max(f"w{w}").cast("double").alias(C.rain_column(w))
                for w in C.RAIN_WINDOWS_MINUTES
            ),
            F.count("*").cast("int").alias("rain_gauges"),
        )
    )
    forecasts = spark.read.table("area_forecasts").select(
        "area",
        seconds("ts").alias("tick_s"),
        "forecast_rain_class",
        "forecast_age_min",
    )
    weather = spark.read.table("tick_weather").select(
        seconds("ts").alias("tick_s"),
        "air_temp_c",
        "rel_humidity_pct",
        "wind_u_kn",
        "wind_v_kn",
        "lightning_records",
    )
    lightning = spark.read.table("area_lightning").select(
        "area", seconds("ts").alias("tick_s"), *LIGHTNING
    )
    keys = ["area", "tick_s"]
    rows = (
        per_area.join(forecasts, keys, "left")
        .join(spark.read.table("area_nearby_rain"), keys, "left")
        .join(weather, "tick_s", "left")
        .join(lightning, keys, "left")
    )
    # upwind needs a wind to look into; lightning needs the feed to have been up for the whole window
    speed, _ = _wind()
    lightning_up = (F.col("tick_s") >= sgt_midnight_s(C.LIGHTNING_START)) & (
        F.col("lightning_records") == C.LIGHTNING_RECORDS_PER_WINDOW
    )
    return rows.select(
        "area",
        F.timestamp_seconds("tick_s").alias("ts"),
        *(C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES),
        "rain_gauges",
        "forecast_rain_class",
        "forecast_age_min",
        F.coalesce(F.col("nearby"), F.lit(0))
        .cast("double")
        .alias("rain_nearby_30min_mm"),
        F.when(speed >= C.UPWIND_MIN_WIND_KN, F.coalesce(F.col("upwind"), F.lit(0)))
        .cast("double")
        .alias("rain_upwind_30min_mm"),
        "air_temp_c",
        "rel_humidity_pct",
        "wind_u_kn",
        "wind_v_kn",
        *(
            F.when(lightning_up, F.coalesce(F.col(c), F.lit(0))).cast("int").alias(c)
            for c in LIGHTNING
        ),
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
