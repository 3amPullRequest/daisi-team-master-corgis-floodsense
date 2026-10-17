"""Silver static layers from data.gov.sg: URA's Master Plan land use and PUB's flood-prone area.

- silver.land_use: every land-use polygon of the 2019 and 2025 Master Plans, each with the days its plan is in force,
  so a tick reads the plan of its day.
- silver.flood_prone_area: PUB's total flood-prone area per year.

Each comes from the newest version data.gov.sg published (PUB's water-level sensors are in reference.py). Two layers
stop at bronze: SLA's vertical control points carry no heights (only each point's number, status and update time), so
elevation needs a DEM; and the radar frames are only archived for now.

A layer lands in bronze once per version, and silver keeps only the newest version (`_newest`), so a layer is a
snapshot, not a history. For land use, each Master Plan's table is read from its GeoJSON features: the properties
(URA's names contain dots, so they are quoted), the area URA measured, and the polygon kept as a GeoJSON string. The two
plans are stacked, each stamped with the days it is in force (MASTER_PLANS in utilities/silver_sources.py), so a
consumer picks the plan of a day by comparing it with valid_from and valid_to. The flood-prone area is a short yearly
table of hectares from the newest version of the CSV. Rows that don't parse, or lack a polygon or a positive area,
are dropped.
"""

from functools import reduce

from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from utilities.silver import get, landed, schema
from utilities.silver_sources import MASTER_PLANS

SILVER = schema(spark)
PLANS = landed(spark, *MASTER_PLANS)


def _newest(rows: DataFrame) -> DataFrame:
    """The rows of the newest `version` a layer landed in."""
    newest = rows.agg(F.max("version").alias("version"))
    return rows.join(newest, "version")


def _property(name: str, type_: str):
    """A GeoJSON feature property; URA's names hold dots (SHAPE.AREA), so they are quoted."""
    return F.expr(
        f"""try_variant_get(payload, "$.feature.properties['{name}']", '{type_}')"""
    )


if PLANS:

    @dp.materialized_view(
        name=f"{SILVER}.land_use",
        comment="URA Master Plan land use, one row per polygon of the 2019 and 2025 plans: land_use (URA's LU_DESC, "
        "e.g. RESIDENTIAL, ROAD, OPEN SPACE, WATERBODY; 33 in all), the area URA measured (area_m2, in SVY21), and "
        "the polygon as GeoJSON (WGS84). valid_from and valid_to are the days its plan is in force (valid_to is the "
        "first day it no longer is): MP2019 until 30 Nov 2025, MP2025 from 1 Dec 2025, so a tick reads the plan of "
        "its day. Zoning is planned use, not land cover.",
    )
    @dp.expect_all_or_drop(
        {
            "described": "land_use IS NOT NULL",
            "polygon": "geometry_type IN ('Polygon', 'MultiPolygon') AND geometry IS NOT NULL",
            "measured": "area_m2 > 0",
        }
    )
    def land_use():
        plans = []
        for table in PLANS:
            plan, valid_from, valid_to = MASTER_PLANS[table]
            rows = spark.read.table(table).select(
                get("version", "string").alias("version"),
                _property("OBJECTID", "bigint").alias("object_id"),
                _property("LU_DESC", "string").alias("land_use"),
                # URA writes a blank as a space
                F.nullif(F.trim(_property("LU_TEXT", "string")), F.lit("")).alias(
                    "land_use_text"
                ),
                _property("GPR", "string").alias("gross_plot_ratio"),
                _property("SHAPE.AREA", "double").alias("area_m2"),
                get("feature.geometry.type", "string").alias("geometry_type"),
                F.to_json(get("feature.geometry", "variant")).alias("geometry"),
            )
            plans.append(
                _newest(rows).select(
                    F.lit(plan).alias("plan"),
                    "object_id",
                    "land_use",
                    "land_use_text",
                    "gross_plot_ratio",
                    "area_m2",
                    "geometry_type",
                    "geometry",
                    F.lit(valid_from).alias("valid_from"),
                    F.lit(valid_to).cast("date").alias("valid_to"),
                    "version",
                )
            )
        return reduce(DataFrame.unionByName, plans)


if landed(spark, "static_flood_prone_area_ha"):

    @dp.materialized_view(
        name=f"{SILVER}.flood_prone_area",
        comment="PUB's total flood-prone area in hectares, by year (27 in 2022 down to 23.3 in 2025), from the newest "
        "version of its data.gov.sg dataset. The 36 places PUB lists as flood-prone are in a PDF, not here.",
    )
    @dp.expect_all_or_drop({"parsed": "year IS NOT NULL AND hectares > 0"})
    def flood_prone_area():
        rows = spark.read.table("static_flood_prone_area_ha").select(
            F.regexp_extract("source_file", r"version=([^/]+)", 1).alias("version"),
            F.expr("try_cast(Year AS INT)").alias("year"),
            F.expr("try_cast(Hectares AS DOUBLE)").alias("hectares"),
        )
        return _newest(rows).select("year", "hectares", "version")
