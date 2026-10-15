"""Silver reference tables: NEA's 47 forecast areas, the rain gauges, and PUB's drain water-level sensors.

Each point comes from the live API pages where they have landed, else from the yearly CSVs. The CSV halves are private
materialized views: they rescan the CSVs (27 million rainfall rows) only when a new yearly file lands, not on every
5-minute update.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.silver import get, in_singapore, landed, schema, svy21_to_wgs84

SILVER = schema(spark)
DAY = F.to_date(F.regexp_extract("source_file", r"date=(\d{4}-\d{2}-\d{2})", 1))

FORECAST_TABLES = landed(spark, "nea_two_hr_forecast", "nea_two_hr_forecast_hist")
RAIN_TABLES = landed(spark, "nea_rainfall", "nea_rainfall_hist")


def _latest_point(points, key: str, extra: list[str]):
    """One row per `key`: the most recently seen point, with the first and last day it was seen."""
    return points.groupBy(key).agg(
        F.max_by("lat", "last_day").alias("lat"),
        F.max_by("lon", "last_day").alias("lon"),
        *(F.max_by(c, "last_day").alias(c) for c in extra),
        F.min("first_day").alias("first_day"),
        F.max("last_day").alias("last_day"),
        F.array_sort(F.collect_set("source")).alias("sources"),
    )


# --- NEA forecast areas ------------------------------------------------------------------------------------------------

if "nea_two_hr_forecast_hist" in FORECAST_TABLES:

    @dp.materialized_view(name="forecast_area_points_hist", private=True)
    def forecast_area_points_hist():
        rows = spark.read.table("nea_two_hr_forecast_hist")
        return rows.groupBy(F.col("location_name").alias("area")).agg(
            F.max_by(F.expr("try_cast(location_latitude AS DOUBLE)"), "date").alias(
                "lat"
            ),
            F.max_by(F.expr("try_cast(location_longitude AS DOUBLE)"), "date").alias(
                "lon"
            ),
            F.min(F.to_date("date")).alias("first_day"),
            F.max(F.to_date("date")).alias("last_day"),
            F.lit("csv").alias("source"),
        )


if FORECAST_TABLES:

    @dp.materialized_view(
        name=f"{SILVER}.forecast_areas",
        comment="NEA's 47 2-hour forecast areas and their label points (WGS84). An alert belongs to the area whose "
        "point is nearest it; the app's map and the per-area joins use them.",
    )
    @dp.expect_all_or_drop(
        {"named": "area IS NOT NULL", "point_in_singapore": in_singapore()}
    )
    def forecast_areas():
        parts = []
        if "nea_two_hr_forecast" in FORECAST_TABLES:
            meta = spark.read.table("nea_two_hr_forecast").select(
                DAY.alias("day"),
                F.explode(
                    get(
                        "body.data.area_metadata",
                        "array<struct<name: string, label_location: struct<latitude: double, longitude: double>>>",
                    )
                ).alias("m"),
            )
            parts.append(
                meta.groupBy(F.col("m.name").alias("area")).agg(
                    F.max_by("m.label_location.latitude", "day").alias("lat"),
                    F.max_by("m.label_location.longitude", "day").alias("lon"),
                    F.min("day").alias("first_day"),
                    F.max("day").alias("last_day"),
                    F.lit("api").alias("source"),
                )
            )
        if "nea_two_hr_forecast_hist" in FORECAST_TABLES:
            parts.append(spark.read.table("forecast_area_points_hist"))
        points = parts[0] if len(parts) == 1 else parts[0].unionByName(parts[1])
        return _latest_point(points, "area", [])


# --- NEA rain gauges ---------------------------------------------------------------------------------------------------

if "nea_rainfall_hist" in RAIN_TABLES:

    @dp.materialized_view(name="rain_stations_hist", private=True)
    def rain_stations_hist():
        rows = spark.read.table("nea_rainfall_hist")
        return rows.groupBy("station_id").agg(
            F.max_by("station_name", "date").alias("name"),
            F.max_by(F.expr("try_cast(location_latitude AS DOUBLE)"), "date").alias(
                "lat"
            ),
            F.max_by(F.expr("try_cast(location_longitude AS DOUBLE)"), "date").alias(
                "lon"
            ),
            F.min(F.to_date("date")).alias("first_day"),
            F.max(F.to_date("date")).alias("last_day"),
            F.lit("csv").alias("source"),
        )


if RAIN_TABLES:

    @dp.materialized_view(
        name=f"{SILVER}.rain_stations",
        comment="NEA rain gauges: location (WGS84), name, and the first and last day each reported. Gauges come and "
        "go (89 in the API's list in Oct 2026, 80 in the 2021-2024 CSVs), but none has moved.",
    )
    @dp.expect_all_or_drop(
        {"identified": "station_id IS NOT NULL", "point_in_singapore": in_singapore()}
    )
    def rain_stations():
        parts = []
        if "nea_rainfall" in RAIN_TABLES:
            meta = spark.read.table("nea_rainfall").select(
                DAY.alias("day"),
                F.explode(
                    get(
                        "body.data.stations",
                        "array<struct<id: string, name: string, location: struct<latitude: double, longitude: double>>>",
                    )
                ).alias("s"),
            )
            parts.append(
                meta.groupBy(F.col("s.id").alias("station_id")).agg(
                    F.max_by("s.name", "day").alias("name"),
                    F.max_by("s.location.latitude", "day").alias("lat"),
                    F.max_by("s.location.longitude", "day").alias("lon"),
                    F.min("day").alias("first_day"),
                    F.max("day").alias("last_day"),
                    F.lit("api").alias("source"),
                )
            )
        if "nea_rainfall_hist" in RAIN_TABLES:
            parts.append(spark.read.table("rain_stations_hist"))
        points = parts[0] if len(parts) == 1 else parts[0].unionByName(parts[1])
        return _latest_point(points, "station_id", ["name"])


# --- PUB drain water-level sensors -------------------------------------------------------------------------------------

if landed(spark, "static_pub_water_level_sensors"):

    @dp.materialized_view(
        name=f"{SILVER}.water_level_sensors",
        comment="PUB's drain water-level sensors, which raise the 90% warnings: name and location, converted from "
        "SVY21 to WGS84. From the newest version of the list only (a snapshot, so sensors removed earlier are absent).",
    )
    @dp.expect_all_or_drop(
        {"identified": "sensor_id IS NOT NULL", "point_in_singapore": in_singapore()}
    )
    def water_level_sensors():
        rows = spark.read.table("static_pub_water_level_sensors").select(
            get("version", "string").alias("version"),
            F.expr(
                """try_variant_get(payload, "$.row['Station ID']", 'string')"""
            ).alias("sensor_id"),
            F.expr(
                """try_variant_get(payload, "$.row['Station Name']", 'string')"""
            ).alias("name"),
            F.expr(
                "try_cast(try_variant_get(payload, '$.row.X', 'string') AS DOUBLE)"
            ).alias("x_svy21"),
            F.expr(
                "try_cast(try_variant_get(payload, '$.row.Y', 'string') AS DOUBLE)"
            ).alias("y_svy21"),
        )
        newest = rows.agg(F.max("version").alias("version"))
        current = rows.join(newest, "version").where(
            "x_svy21 IS NOT NULL AND y_svy21 IS NOT NULL"
        )
        point = svy21_to_wgs84("y_svy21", "x_svy21")
        return current.select(
            "sensor_id",
            "name",
            point["lat"].alias("lat"),
            point["lon"].alias("lon"),
            "x_svy21",
            "y_svy21",
            "version",
        )
