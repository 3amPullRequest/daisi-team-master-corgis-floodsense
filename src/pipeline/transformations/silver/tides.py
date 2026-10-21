"""Silver tide tables: UHSLC's hourly sea level at Tanjong Pagar (1984-2023) and NEA's high and low waters (the
current and next month, scraped daily).

Both are materialized views: the UHSLC file lands once, and NEA's table is a few hundred rows a month. Gold fits the
tide model on the first, checks its datum against the second, and fills every tick with the two (gold/tides.py).
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.silver import get, landed, schema

from daisi_team_master_corgis_floodsense.tides import MISSING_MM

SILVER = schema(spark)
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")  # fmt: skip
MONTHS = F.create_map(
    *(F.lit(x) for i, m in enumerate(MONTH_NAMES) for x in (m, i + 1))
)
# Tanjong Pagar's lowest and highest readings in 1984-2023 were 0.02 m and 3.67 m above UHSLC's zero.
PLAUSIBLE_M = "BETWEEN -0.5 AND 4.5"


def _int(column: str):
    return F.expr(f"try_cast({column} AS INT)")


if landed(spark, "uhslc_tanjong_pagar_hourly"):

    @dp.materialized_view(
        name=f"{SILVER}.tide_gauge_hourly",
        comment="UHSLC research-quality hourly sea level at Tanjong Pagar (station 699), one row per hour, 1984-2023. "
        "sea_level_m is metres above UHSLC's station zero (0.005-0.012 m below NEA's Chart Datum, gold.tide_datum), "
        f"null where UHSLC marks the hour missing ({MISSING_MM}). ts is UHSLC's GMT hour.",
        cluster_by=["ts"],
    )
    @dp.expect_all_or_drop({"hour_parsed": "ts IS NOT NULL"})
    @dp.expect(
        "sea_level_plausible", f"sea_level_m IS NULL OR sea_level_m {PLAUSIBLE_M}"
    )
    def tide_gauge_hourly():
        rows = spark.read.table("uhslc_tanjong_pagar_hourly")
        stamp = F.format_string(
            "%04d-%02d-%02dT%02d:00:00Z",
            _int("year"),
            _int("month"),
            _int("day"),
            _int("hour"),
        )
        mm = _int("sea_level_mm")
        readings = rows.withColumn("stamp", stamp).select(
            F.expr("try_cast(stamp AS TIMESTAMP)").alias("ts"),
            F.when(mm != MISSING_MM, mm / 1000).alias("sea_level_m"),
            "file_modified_at",
            "source_file",
        )
        # a re-download replaces the old file's hours
        return readings.groupBy("ts").agg(
            F.max_by("sea_level_m", "file_modified_at").alias("sea_level_m"),
            F.max_by("source_file", "file_modified_at").alias("source_file"),
        )


if landed(spark, "nea_tide_timings"):

    @dp.materialized_view(
        name=f"{SILVER}.tide_extremes",
        comment="NEA's predicted high and low waters (MPA's tide table for Singapore), metres above Chart Datum, one "
        "row per high or low water. Each month as NEA last published it: the newest scrape of a month replaces the "
        "older ones. ts is the time NEA prints (SGT), to the minute.",
        cluster_by=["ts"],
    )
    @dp.expect_all_or_drop(
        {
            "tide_parsed": "ts IS NOT NULL AND kind IS NOT NULL AND height_m IS NOT NULL",
            "height_plausible": f"height_m {PLAUSIBLE_M}",
        }
    )
    def tide_extremes():
        rows = spark.read.table("nea_tide_timings").select(
            get("month_label", "string").alias("month_label"),  # 'Oct 2026'
            get("day", "string").alias("day"),
            get("time", "string").alias("time"),  # 'HHMM', SGT
            get("height_m", "string").alias("height_text"),
            get("hl", "string").alias("hl"),
            F.coalesce("fetched_at", "file_modified_at").alias("fetched_at"),
            "source_file",
        )
        newest = rows.withColumn(
            "newest", F.max("fetched_at").over(Window.partitionBy("month_label"))
        ).where(F.col("fetched_at") == F.col("newest"))
        year = _int("substring(month_label, -4, 4)")
        month = F.try_element_at(MONTHS, F.substring("month_label", 1, 3))
        stamp = F.format_string(
            "%04d-%02d-%02dT%s:%s:00+08:00",
            year,
            month,
            _int("day"),
            F.substring("time", 1, 2),
            F.substring("time", 3, 2),
        )
        return (
            newest.withColumn("stamp", stamp)
            .select(
                F.expr("try_cast(stamp AS TIMESTAMP)").alias("ts"),
                F.when(F.upper("hl") == "H", "high")
                .when(F.upper("hl") == "L", "low")
                .alias("kind"),
                F.expr("try_cast(height_text AS DOUBLE)").alias("height_m"),
                F.make_date(year, month, F.lit(1)).alias("month"),
                "fetched_at",
                "source_file",
            )
            .dropDuplicates(["ts", "kind"])
        )
