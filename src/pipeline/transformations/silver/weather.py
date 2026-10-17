"""silver.weather_1min: NEA's 1-minute weather readings (air temperature, relative humidity, wind speed, wind
direction) at its automatic weather stations, one per station, variable and minute, from the yearly CSVs (2021-2024),
the API day backfill and the live ticks. The stations themselves are silver.weather_stations (reference.py).

As with rainfall, the same reading lands many times: every live tick re-fetches the newest 25 minutes and the daily
repair re-pulls the day. Auto CDC keeps the latest fetch of each, and each update merges only the pages that landed
since the last one. A yearly CSV also holds a few readings twice, revised minutes after NEA first published them (42 of
28 million temperatures, 46 humidities); the later publication wins.

Readings stay at NEA's resolution, long rather than wide, so the four variables merge independently. Wind speed and
direction are already NEA's 10-minute means, reported every minute. What a 5-minute tick may use (the reading lag,
published_at) is gold's business, as for rain.

One target takes up to eight flows: for each of the four variables, an API view and a CSV view (WEATHER_VARIABLES in
utilities/silver_sources.py says which bronze tables). Each view reads its source into the same shape, tags the rows
with the variable, and drops readings that don't parse or fall outside that variable's plausible range. The flows
merge by station, variable and minute. A reading is ordered by when and from which file it was fetched and then by its
publication time, so that a CSV revision, which shares a file with the reading it revises, replaces it. API rows carry
no publication time (NULL), the CSVs do.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.silver import get, landed, schema
from utilities.silver_sources import WEATHER_VARIABLES

SILVER = schema(spark)
TARGET = f"{SILVER}.weather_1min"
SOURCES = landed(
    spark, *(t for api, _ in WEATHER_VARIABLES.values() for t in (api, f"{api}_hist"))
)


def _checks(variable: str) -> dict[str, str]:
    low, high = WEATHER_VARIABLES[variable][1]
    return {
        "reading_parsed": "station_id IS NOT NULL AND ts IS NOT NULL AND value IS NOT NULL",
        "value_plausible": f"value BETWEEN {low} AND {high}",
    }


def _api_readings(variable: str, table: str) -> str:
    """Every reading on every API page of `table` that answered 200 (a 404 or an empty page only marks a day as done)."""
    name = f"weather_{variable}_api_readings"

    @dp.temporary_view(name=name)
    @dp.expect_all_or_drop(_checks(variable))
    def api_readings():
        pages = spark.readStream.table(table).where(get("status", "int") == 200)
        readings = pages.select(
            # never NULL, so its type matches the CSV flows' sequence_by (Auto CDC compares nullability too)
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
            ).alias("v"),
        )
        return values.select(
            F.col("v.stationId").alias("station_id"),
            F.lit(variable).alias("variable"),
            F.expr("try_cast(ts_text AS TIMESTAMP)").alias("ts"),
            F.col("v.value").alias("value"),
            F.lit(None).cast("timestamp").alias("published_at"),
            "fetched_at",
            F.lit("api").alias("source"),
            "source_file",
        )

    return name


def _csv_readings(variable: str, table: str) -> str:
    """The yearly CSVs. They say when NEA published each reading, which the API doesn't."""
    name = f"weather_{variable}_csv_readings"

    @dp.temporary_view(name=name)
    @dp.expect_all_or_drop(_checks(variable))
    def csv_readings():
        rows = spark.readStream.table(table)
        return rows.select(
            "station_id",
            F.lit(variable).alias("variable"),
            F.expr("try_cast(timestamp AS TIMESTAMP)").alias("ts"),
            F.expr("try_cast(reading_value AS DOUBLE)").alias("value"),
            F.expr("try_cast(reading_update_timestamp AS TIMESTAMP)").alias(
                "published_at"
            ),
            F.col("file_modified_at").alias("fetched_at"),
            F.lit("csv").alias("source"),
            "source_file",
        )

    return name


if SOURCES:
    dp.create_streaming_table(
        name=TARGET,
        comment="NEA weather stations' 1-minute readings, one row per station, variable and minute: the latest fetch "
        "of each. variable is air_temp_c (deg C), rel_humidity_pct (%), wind_speed_kn (knots) or wind_dir_deg "
        "(degrees the wind blows from); both wind variables are NEA's 10-minute means, reported every minute. ts is "
        "NEA's reading timestamp; published_at, when NEA published it, comes with the yearly CSVs only. source is "
        "'csv' (2021-2024) or 'api' (2025 on: day backfill and live ticks).",
        cluster_by=["ts"],
    )
    for variable, (api, _) in WEATHER_VARIABLES.items():
        for table, view in (
            (api, _api_readings),
            (f"{api}_hist", _csv_readings),
        ):
            if table in SOURCES:
                source = view(variable, table)
                dp.create_auto_cdc_flow(
                    target=TARGET,
                    source=source,
                    keys=["station_id", "variable", "ts"],
                    # a CSV revision shares its file with the reading it revises, so publication breaks the tie
                    sequence_by=F.struct(
                        "fetched_at",
                        "source_file",
                        F.coalesce("published_at", "fetched_at").alias("published"),
                    ),
                    stored_as_scd_type=1,
                    name=f"{source}_into_weather_1min",
                )
