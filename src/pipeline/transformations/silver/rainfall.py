"""silver.rainfall_5min: one 5-minute rainfall total per gauge and timestamp, from the yearly CSVs (2021-2024), the API
day backfill and the live ticks.

The same reading lands many times: every live tick re-fetches the last two hours, the daily repair re-pulls the day,
and a timestamp's gauges can arrive over several ticks. Auto CDC keeps one row per (station_id, ts), the latest
fetched, and each update merges only the pages that landed since the last one.

Two sources feed one table, each through its own view and its own Auto CDC flow. The API view reads the pages that
answered 200 (a 404 or an empty page only marks a day as done) and explodes them to one row per gauge and timestamp.
The CSV view reads the yearly files, which carry one thing the API doesn't: when NEA published each reading
(`published_at`, from its update timestamp), which is NULL on API rows. Gold needs it, because a reading counts at a
tick only once NEA had published it. Both views drop readings that don't parse or that are implausible (outside 0-60 mm
in 5 minutes). The flows order competing copies of a reading by when and from which file they were fetched, so the
latest fetch wins.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.silver import get, landed, schema

SILVER = schema(spark)
TARGET = f"{SILVER}.rainfall_5min"
SOURCES = landed(spark, "nea_rainfall", "nea_rainfall_hist")

# The largest 5-minute total in the 2021-2024 CSVs (27 million readings) is 37.2 mm.
READING_CHECKS = {
    "reading_parsed": "station_id IS NOT NULL AND ts IS NOT NULL AND rain_mm IS NOT NULL",
    "rain_plausible": "rain_mm BETWEEN 0 AND 60",
}

if "nea_rainfall" in SOURCES:

    @dp.temporary_view(name="rainfall_api_readings")
    @dp.expect_all_or_drop(READING_CHECKS)
    def rainfall_api_readings():
        """Every reading on every API page that answered 200 (a 404 or an empty page only marks a day as done)."""
        pages = spark.readStream.table("nea_rainfall").where(
            get("status", "int") == 200
        )
        readings = pages.select(
            # never NULL, so its type matches the CSV flow's sequence_by (Auto CDC compares nullability too)
            F.coalesce("fetched_at", "file_modified_at").alias("fetched_at"),
            "source_file",
            F.explode(get("body.data.readings", "array<variant>")).alias("reading"),
        )
        values = readings.select(
            "fetched_at",
            "source_file",
            get("timestamp", "string", "reading").alias("ts_text"),
            F.explode(
                get(
                    "data", "array<struct<stationId: string, value: double>>", "reading"
                )
            ).alias("value"),
        )
        return values.select(
            F.col("value.stationId").alias("station_id"),
            F.expr("try_cast(ts_text AS TIMESTAMP)").alias("ts"),
            F.col("value.value").alias("rain_mm"),
            F.lit(None).cast("timestamp").alias("published_at"),
            "fetched_at",
            F.lit("api").alias("source"),
            "source_file",
        )


if "nea_rainfall_hist" in SOURCES:

    @dp.temporary_view(name="rainfall_csv_readings")
    @dp.expect_all_or_drop(READING_CHECKS)
    def rainfall_csv_readings():
        """The yearly CSVs. They say when NEA published each reading, which the API doesn't."""
        rows = spark.readStream.table("nea_rainfall_hist")
        return rows.select(
            "station_id",
            F.expr("try_cast(timestamp AS TIMESTAMP)").alias("ts"),
            F.expr("try_cast(reading_value AS DOUBLE)").alias("rain_mm"),
            F.expr("try_cast(reading_update_timestamp AS TIMESTAMP)").alias(
                "published_at"
            ),
            F.col("file_modified_at").alias("fetched_at"),
            F.lit("csv").alias("source"),
            "source_file",
        )


if SOURCES:
    dp.create_streaming_table(
        name=TARGET,
        comment="NEA rain gauges' 5-minute rainfall totals (mm), one row per gauge and timestamp: the latest fetch of "
        "each. ts is NEA's reading timestamp; published_at, when NEA published it, comes with the yearly CSVs only. "
        "source is 'csv' (2021-2024) or 'api' (2025 on: day backfill and live ticks).",
        cluster_by=["ts"],
    )
    for name, view in (
        ("nea_rainfall", "rainfall_api_readings"),
        ("nea_rainfall_hist", "rainfall_csv_readings"),
    ):
        if name in SOURCES:
            dp.create_auto_cdc_flow(
                target=TARGET,
                source=view,
                keys=["station_id", "ts"],
                sequence_by=F.struct("fetched_at", "source_file"),
                stored_as_scd_type=1,
                name=f"{view}_into_rainfall_5min",
            )
