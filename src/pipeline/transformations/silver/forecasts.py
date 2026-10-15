"""silver.forecasts_2h: NEA's 2-hour forecast for each area, every issue of it, from the yearly CSVs (2021-2024), the
API day backfill and the live ticks.

NEA issues a forecast about every 30 minutes and amends it in between, so a forecast is keyed by its area, its nominal
issue time and the moment it was published (updated_at). A feature at tick t may only use forecasts with
updated_at <= t. Like rainfall, Auto CDC keeps the latest fetch of each and merges only new pages.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.forecast_categories import FORECASTS
from utilities.silver import get, get_ts, landed, schema

SILVER = schema(spark)
TARGET = f"{SILVER}.forecasts_2h"
SOURCES = landed(spark, "nea_two_hr_forecast", "nea_two_hr_forecast_hist")

FORECAST_CHECKS = {
    "forecast_parsed": "area IS NOT NULL AND issued_at IS NOT NULL AND updated_at IS NOT NULL AND forecast IS NOT NULL",
}
CODE = F.create_map(
    *(F.lit(x) for text, (code, _) in FORECASTS.items() for x in (text, code))
)
RAIN_CLASS = F.create_map(
    *(F.lit(x) for text, (_, rain) in FORECASTS.items() for x in (text, rain))
)


def _classified(rows):
    """rain_class from the category text; NEA's code too, for the API, which sends only the text. A category not in
    FORECASTS gets NULLs (try_element_at: ANSI mode would fail the update on a missing key)."""
    return rows.withColumn(
        "forecast_code",
        F.coalesce("forecast_code", F.try_element_at(CODE, F.col("forecast"))),
    ).withColumn("rain_class", F.try_element_at(RAIN_CLASS, F.col("forecast")))


if "nea_two_hr_forecast" in SOURCES:

    @dp.temporary_view(name="forecast_api_rows")
    @dp.expect_all_or_drop(FORECAST_CHECKS)
    @dp.expect("category_known", "rain_class IS NOT NULL")
    def forecast_api_rows():
        pages = spark.readStream.table("nea_two_hr_forecast").where(
            get("status", "int") == 200
        )
        items = pages.select(
            # never NULL, so its type matches the CSV flow's sequence_by (Auto CDC compares nullability too)
            F.coalesce("fetched_at", "file_modified_at").alias("fetched_at"),
            "source_file",
            F.explode(get("body.data.items", "array<variant>")).alias("item"),
        )
        rows = items.select(
            get_ts("timestamp", "item").alias("issued_at"),
            get_ts("update_timestamp", "item").alias("updated_at"),
            get_ts("valid_period.start", "item").alias("valid_from"),
            get_ts("valid_period.end", "item").alias("valid_to"),
            F.explode(
                get(
                    "forecasts", "array<struct<area: string, forecast: string>>", "item"
                )
            ).alias("f"),
            "fetched_at",
            "source_file",
        )
        return _classified(
            rows.select(
                F.col("f.area").alias("area"),
                "issued_at",
                "updated_at",
                "valid_from",
                "valid_to",
                F.col("f.forecast").alias("forecast"),
                F.lit(None).cast("string").alias("forecast_code"),
                "fetched_at",
                F.lit("api").alias("source"),
                "source_file",
            )
        )


if "nea_two_hr_forecast_hist" in SOURCES:

    @dp.temporary_view(name="forecast_csv_rows")
    @dp.expect_all_or_drop(FORECAST_CHECKS)
    @dp.expect("category_known", "rain_class IS NOT NULL")
    def forecast_csv_rows():
        rows = spark.readStream.table("nea_two_hr_forecast_hist")
        return _classified(
            rows.select(
                F.col("location_name").alias("area"),
                F.expr("try_cast(timestamp AS TIMESTAMP)").alias("issued_at"),
                F.expr("try_cast(update_timestamp AS TIMESTAMP)").alias("updated_at"),
                F.expr("try_cast(valid_period_start AS TIMESTAMP)").alias("valid_from"),
                F.expr("try_cast(valid_period_end AS TIMESTAMP)").alias("valid_to"),
                F.col("forecast_text").alias("forecast"),
                "forecast_code",
                F.col("file_modified_at").alias("fetched_at"),
                F.lit("csv").alias("source"),
                "source_file",
            )
        )


if SOURCES:
    dp.create_streaming_table(
        name=TARGET,
        comment="NEA 2-hour forecasts per area, every issue and amendment: issued_at is the nominal issue time, "
        "updated_at when it was published, valid_from/valid_to the period it covers. rain_class orders the categories "
        "by forecast rain (0 none, 1 light, 2 moderate, 3 heavy, 4 thundery, 5 heavy thundery); NULL for a category "
        "utilities/forecast_categories.py doesn't know yet.",
        cluster_by=["updated_at"],
    )
    for name, view in (
        ("nea_two_hr_forecast", "forecast_api_rows"),
        ("nea_two_hr_forecast_hist", "forecast_csv_rows"),
    ):
        if name in SOURCES:
            dp.create_auto_cdc_flow(
                target=TARGET,
                source=view,
                keys=["area", "issued_at", "updated_at"],
                sequence_by=F.struct("fetched_at", "source_file"),
                stored_as_scd_type=1,
                name=f"{view}_into_forecasts_2h",
            )
