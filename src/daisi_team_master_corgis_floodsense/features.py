"""The feature definitions of gold.area_forecasts and gold.area_features, as Spark DataFrame functions shared by the
flood-gold pipeline (every tick since 2021) and flood-live's score task (the newest ticks). One definition, so a live
prediction reads exactly what the models were trained on.

What gold.area_features is built from, in the order the functions here are applied:

- `forecast_ticks`: the 2-hour forecast valid at each tick (gold.area_forecasts).
- `gauge_rain`: per gauge and tick, the rain totals over each window (gold.gauge_rain).
- `tick_weather`: the island-wide weather at each tick, and how complete the lightning feed was (gold.tick_weather).
- `nearby_rain`: the rain 5-20 km from each area, and upwind of it (gold.area_nearby_rain); it reads the wind.
- `area_features`: all of them, and the tide, per area and tick.

Lightning (gold.area_lightning) stays in the gold pipeline: it is held back from the models (contract.HELD_BACK_FEATURES),
so live scoring leaves its columns null.

Times are worked in epoch seconds, so no result depends on the session time zone. SGT has kept UTC+8 since 1982, so an
SGT midnight is the UTC one minus 8 hours, and the 5-minute tick grid is the same in both.
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .ml import contract as C

TICK_S = C.TICK_MINUTES * 60
LAG_S = C.READING_LAG_MINUTES * 60
SLOTS = max(C.RAIN_WINDOWS_MINUTES) // C.TICK_MINUTES  # readings in the longest window
# how many ticks' windows hold a weather reading
WEATHER_SLOTS = C.WEATHER_MINUTES // C.TICK_MINUTES
LIGHTNING_S = C.LIGHTNING_MINUTES * 60
# how far back a tick's inputs reach: its longest rain window ends READING_LAG before it, its weather window likewise,
# and a 2-hour forecast is valid at most 2 hours after its publication
RAIN_REACH_S = LAG_S + SLOTS * TICK_S
WEATHER_REACH_S = LAG_S + WEATHER_SLOTS * TICK_S
FORECAST_REACH_S = 2 * 3600
NEARBY = f"w{C.NEARBY_RAIN_MINUTES}"  # the gauge window the nearby and upwind rain read
LIGHTNING = tuple(C.lightning_column(km) for km in C.LIGHTNING_RADII_KM)
# the SGT midnight lightning starts, in epoch seconds
LIGHTNING_START_S = (C.LIGHTNING_START - date(1970, 1, 1)).days * 86400 - 8 * 3600
EXACT = "decimal(10,4)"  # weather means are of exact decimals, so they don't depend on how Spark splits the work


def seconds(ts: str) -> Column:
    """Epoch seconds of a TIMESTAMP column."""
    return F.unix_timestamp(ts)


def tick_at_or_after(s: Column) -> Column:
    return F.ceil(s / TICK_S) * TICK_S


def tick_after(s: Column) -> Column:
    return F.floor(s / TICK_S) * TICK_S + TICK_S


def tick_before(s: Column) -> Column:
    return F.ceil(s / TICK_S) * TICK_S - TICK_S


def ticks_between(first: Column, last: Column) -> Column:
    """One row per tick from `first` to `last` (epoch seconds on the grid, both included); none if last < first."""
    step = F.lit(TICK_S).cast("bigint")
    return F.explode(
        F.when(
            last >= first, F.sequence(first.cast("bigint"), last.cast("bigint"), step)
        )
    )


def distance_km(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    """Haversine, as locations.haversine_km."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    h = F.pow(F.sin((p2 - p1) / 2), 2) + F.cos(p1) * F.cos(p2) * F.pow(
        F.sin(F.radians(lon2 - lon1) / 2), 2
    )
    return F.lit(2 * 6371.0088) * F.asin(F.sqrt(h))


def bearing_deg(lat1: str, lon1: str, lat2: str, lon2: str) -> Column:
    """Initial bearing from point 1 to point 2 (columns named), degrees clockwise from north."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dl = F.radians(F.col(lon2) - F.col(lon1))
    east = F.sin(dl) * F.cos(p2)
    north = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dl)
    return F.pmod(F.degrees(F.atan2(east, north)), F.lit(360.0))


def wind(u: str = "wind_u_kn", v: str = "wind_v_kn") -> tuple[Column, Column]:
    """(speed in knots, the direction it blows from in degrees) of a wind vector."""
    speed = F.sqrt(F.col(u) * F.col(u) + F.col(v) * F.col(v))
    blowing_from = F.pmod(F.degrees(F.atan2(F.col(u), F.col(v))) + 180, F.lit(360.0))
    return speed, blowing_from


def forecast_ticks(forecasts: DataFrame) -> DataFrame:
    """gold.area_forecasts from silver.forecasts_2h rows: each area's newest forecast published (updated_at) by each
    tick, while its valid period lasts, with its rain class and age."""
    published = forecasts.where(
        "area IS NOT NULL AND updated_at IS NOT NULL AND valid_to IS NOT NULL AND rain_class IS NOT NULL"
    ).select(
        "area",
        "rain_class",
        seconds("updated_at").alias("published_s"),
        seconds("issued_at").alias("issued_s"),
        seconds("valid_to").alias("valid_to_s"),
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


def gauge_rain(readings: DataFrame) -> DataFrame:
    """Per gauge and tick (station_id, tick_s), its rain totals over each of RAIN_WINDOWS_MINUTES (w15, w30, w60), from
    silver.rainfall_5min rows. A reading counts at the ticks whose windows hold it, from READING_LAG after its timestamp
    and not before NEA published it where NEA says when: each reading is expanded into those ticks as slot 0, 1, 2 ...,
    so a window's total is a plain group-by and cannot contain a reading its live twin hadn't yet received."""
    readings = readings.where(
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


def tick_weather(
    readings: DataFrame, lightning_records: DataFrame | None = None
) -> DataFrame:
    """gold.tick_weather from silver.weather_1min rows and silver.lightning_records rows: per tick, the mean air
    temperature, humidity and wind (u eastward, v northward, knots) over NEA's stations, each station's newest reading
    in the WEATHER_MINUTES ending READING_LAG before the tick that NEA had published by it; the stations behind each
    mean; and how many of the lightning records of the last LIGHTNING_MINUTES exist (null without `lightning_records`).

    Wind speed and direction are combined into u and v per station before averaging, because directions can't be
    averaged as numbers. NEA gives the direction the wind blows from, so the vector points the other way."""
    readings = readings.where(
        "value IS NOT NULL AND ts IS NOT NULL AND station_id IS NOT NULL"
    )
    # slot k: the reading is in the window of the kth tick from the first whose window holds it
    counted = readings.select(
        "station_id",
        "variable",
        "value",
        seconds("ts").alias("reading_s"),
        seconds("published_at").alias("published_s"),
        F.explode(F.sequence(F.lit(0), F.lit(WEATHER_SLOTS - 1))).alias("slot"),
    ).withColumn(
        "tick_s",
        tick_at_or_after(F.col("reading_s")) + LAG_S + F.col("slot") * TICK_S,
    )
    counted = counted.where(
        F.col("published_s").isNull() | (F.col("published_s") <= F.col("tick_s"))
    )
    newest = counted.groupBy("station_id", "variable", "tick_s").agg(
        F.max_by("value", "reading_s").alias("value")
    )
    per_station = (
        newest.groupBy("station_id", "tick_s")
        .pivot("variable", list(C.WEATHER_READINGS))
        .agg(F.first("value"))
    )
    speed, blowing_from = F.col("wind_speed_kn"), F.radians("wind_dir_deg")
    windy = speed.isNotNull() & blowing_from.isNotNull()
    per_tick = per_station.groupBy("tick_s").agg(
        F.avg(F.col("air_temp_c").cast(EXACT)).cast("double").alias("air_temp_c"),
        F.avg(F.col("rel_humidity_pct").cast(EXACT))
        .cast("double")
        .alias("rel_humidity_pct"),
        F.avg(F.when(windy, -speed * F.sin(blowing_from)).cast(EXACT))
        .cast("double")
        .alias("wind_u_kn"),
        F.avg(F.when(windy, -speed * F.cos(blowing_from)).cast(EXACT))
        .cast("double")
        .alias("wind_v_kn"),
        F.count("air_temp_c").alias("temp_stations"),
        F.count("rel_humidity_pct").alias("humidity_stations"),
        F.count(F.when(windy, 1)).alias("wind_stations"),
    )
    if lightning_records is None:
        rows, records = per_tick, F.lit(None)
    else:
        # a record counts for the ticks whose window (t - LIGHTNING_MINUTES, t] holds it
        record_s = seconds("record_ts")
        counts = (
            lightning_records.where("record_ts IS NOT NULL")
            .select(
                ticks_between(
                    tick_at_or_after(record_s), tick_before(record_s + LIGHTNING_S)
                ).alias("tick_s")
            )
            .groupBy("tick_s")
            .agg(F.count("*").alias("lightning_records"))
        )
        # a full join, so a tick with only lightning records still has a row
        rows = per_tick.join(counts, "tick_s", "full")
        records = F.coalesce("lightning_records", F.lit(0))
    return rows.select(
        F.timestamp_seconds("tick_s").alias("ts"),
        "air_temp_c",
        "rel_humidity_pct",
        "wind_u_kn",
        "wind_v_kn",
        *(
            F.coalesce(c, F.lit(0)).cast("int").alias(c)
            for c in ("temp_stations", "humidity_stations", "wind_stations")
        ),
        records.cast("int").alias("lightning_records"),
    )


def nearby_rain(
    rain: DataFrame, areas: DataFrame, gauges: DataFrame, weather: DataFrame
) -> DataFrame:
    """gold.area_nearby_rain: per area and tick (area, tick_s), the largest NEARBY_RAIN_MINUTES total among the gauges
    NEARBY_RAIN_KM away (`nearby`), and among those of them upwind (`upwind`). From gauge_rain's rows,
    silver.forecast_areas, silver.rain_stations and tick_weather's rows. A gauge is upwind when its bearing from the
    area is within UPWIND_HALF_ANGLE_DEG of the direction the island-wide wind blows from, and the wind is at least
    UPWIND_MIN_WIND_KN. Rows only where one of the gauges rained; area_features reads a missing row as none."""
    areas = areas.select(
        "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
    )
    near, far = C.NEARBY_RAIN_KM
    km = distance_km(F.col("area_lat"), F.col("area_lon"), F.col("lat"), F.col("lon"))
    gauges = gauges.select("station_id", "lat", "lon")
    pairs = (
        areas.crossJoin(gauges)  # 47 areas x ~100 gauges
        .select(
            "area",
            "station_id",
            km.alias("km"),
            bearing_deg("area_lat", "area_lon", "lat", "lon").alias("bearing"),
        )
        .where((F.col("km") > near) & (F.col("km") <= far))
    )
    wet = rain.where(F.col(NEARBY) > 0).select("station_id", "tick_s", NEARBY)
    speed, blowing_from = wind()
    winds = weather.where("wind_u_kn IS NOT NULL").select(
        seconds("ts").alias("tick_s"),
        speed.alias("wind_kn"),
        blowing_from.alias("blowing_from"),
    )
    rows = wet.join(F.broadcast(pairs), "station_id").join(winds, "tick_s", "left")
    off_wind = F.abs(F.pmod(F.col("bearing") - F.col("blowing_from") + 180, 360) - 180)
    upwind = (F.col("wind_kn") >= C.UPWIND_MIN_WIND_KN) & (
        off_wind <= C.UPWIND_HALF_ANGLE_DEG
    )
    return rows.groupBy("area", "tick_s").agg(
        F.max(NEARBY).alias("nearby"),
        F.max(F.when(upwind, F.col(NEARBY))).alias("upwind"),
    )


def area_features(
    rain: DataFrame,
    reach: DataFrame,
    forecasts: DataFrame,
    tide: DataFrame,
    nearby: DataFrame,
    weather: DataFrame,
    lightning: DataFrame | None = None,
) -> DataFrame:
    """gold.area_features (contract.AREA_FEATURES_COLUMNS, in its order) from gauge_rain's rows,
    gold.area_rain_stations, forecast_ticks' rows, gold.tide_levels, nearby_rain's rows, tick_weather's rows and
    gold.area_lightning (or none: its columns are then null). Per area and tick, the largest total among the gauges in
    reach for each window (the max, not the mean, because a flash flood follows the most intense cell), joined to the
    rest. A row exists only where a gauge in reach had a reading in the last hour. The optional columns are null where
    their source had nothing: no lightning before its feed began or when it was down for part of the window, no upwind
    rain without a wind."""
    keys = ["area", "tick_s"]
    per_area = (
        rain.join(
            F.broadcast(
                reach.where("station_id IS NOT NULL").select("area", "station_id")
            ),
            "station_id",
        )
        .groupBy(*keys)
        .agg(
            *(
                F.max(f"w{w}").cast("double").alias(C.rain_column(w))
                for w in C.RAIN_WINDOWS_MINUTES
            ),
            F.count("*").cast("int").alias("rain_gauges"),
        )
    )
    forecast = forecasts.select(
        "area",
        seconds("ts").alias("tick_s"),
        "forecast_rain_class",
        "forecast_age_min",
    )
    weather = weather.select(
        seconds("ts").alias("tick_s"),
        "air_temp_c",
        "rel_humidity_pct",
        "wind_u_kn",
        "wind_v_kn",
        "lightning_records",
    )
    # the same tide for every area: the model's, not the gauge's (contract.py)
    tides = tide.select(
        seconds("ts").alias("tick_s"),
        F.col("predicted_m").alias("tide_height_m"),
        "minutes_to_high_tide",
    )
    rows = (
        per_area.join(forecast, keys, "left")
        .join(nearby, keys, "left")
        .join(weather, "tick_s", "left")
        .join(tides, "tick_s", "left")
    )
    if lightning is None:
        strikes = [F.lit(None).cast("int").alias(c) for c in LIGHTNING]
    else:
        rows = rows.join(
            lightning.select("area", seconds("ts").alias("tick_s"), *LIGHTNING),
            keys,
            "left",
        )
        # lightning needs the feed to have been up for the whole window
        lightning_up = (F.col("tick_s") >= LIGHTNING_START_S) & (
            F.col("lightning_records") == C.LIGHTNING_RECORDS_PER_WINDOW
        )
        strikes = [
            F.when(lightning_up, F.coalesce(F.col(c), F.lit(0))).cast("int").alias(c)
            for c in LIGHTNING
        ]
    # upwind needs a wind to look into
    speed, _ = wind()
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
        *strikes,
        "tide_height_m",
        "minutes_to_high_tide",
    )
