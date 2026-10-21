"""The feature definitions of gold.area_forecasts and gold.area_features, as Spark DataFrame functions shared by the
flood-gold pipeline (every tick since 2021) and flood-live's score task (the newest ticks). One definition, so a live
prediction reads exactly what the models were trained on.

Times are worked in epoch seconds, so no result depends on the session time zone. SGT has kept UTC+8 since 1982, so an
SGT midnight is the UTC one minus 8 hours, and the 5-minute tick grid is the same in both.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .ml import contract as C

TICK_S = C.TICK_MINUTES * 60
LAG_S = C.READING_LAG_MINUTES * 60
SLOTS = max(C.RAIN_WINDOWS_MINUTES) // C.TICK_MINUTES  # readings in the longest window
# how far back a tick's inputs reach: its longest rain window ends READING_LAG before it, and a 2-hour forecast is
# valid at most 2 hours after its publication
RAIN_REACH_S = LAG_S + SLOTS * TICK_S
FORECAST_REACH_S = 2 * 3600


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


def area_features(
    readings: DataFrame, reach: DataFrame, forecasts: DataFrame, tide: DataFrame
) -> DataFrame:
    """gold.area_features (contract.AREA_FEATURES_COLUMNS) from silver.rainfall_5min rows, gold.area_rain_stations,
    forecast_ticks' rows and gold.tide_levels. A reading counts at the ticks whose windows hold it, from READING_LAG
    after its timestamp and not before NEA published it where NEA says when; a row exists only where a gauge in reach
    had a reading in the last hour."""
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
    per_gauge = counted.groupBy("station_id", "tick_s").agg(
        *(
            F.sum(F.when(F.col("slot") < w // C.TICK_MINUTES, F.col("mm"))).alias(
                f"w{w}"
            )
            for w in C.RAIN_WINDOWS_MINUTES
        )
    )
    per_area = (
        per_gauge.join(
            F.broadcast(
                reach.where("station_id IS NOT NULL").select("area", "station_id")
            ),
            "station_id",
        )
        .groupBy("area", "tick_s")
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
    # the same tide for every area: the model's, not the gauge's (contract.py)
    tides = tide.select(
        seconds("ts").alias("tick_s"),
        F.col("predicted_m").alias("tide_height_m"),
        "minutes_to_high_tide",
    )
    return (
        per_area.join(forecast, ["area", "tick_s"], "left")
        .join(tides, "tick_s", "left")
        .select(
            "area",
            F.timestamp_seconds("tick_s").alias("ts"),
            *(C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES),
            "rain_gauges",
            "forecast_rain_class",
            "forecast_age_min",
            "tide_height_m",
            "minutes_to_high_tide",
        )
    )
