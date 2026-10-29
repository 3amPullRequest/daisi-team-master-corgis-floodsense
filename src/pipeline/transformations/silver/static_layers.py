"""Silver static layers from data.gov.sg: URA's Master Plan land use, PUB's flood-prone area and the national map.

- silver.land_use: every land-use polygon of the 2019 and 2025 Master Plans, each with the days its plan is in force,
  so a tick reads the plan of its day.
- silver.flood_prone_area: PUB's total flood-prone area per year.
- silver.map_polygons (25 Oct): the national map's polygons: the coastal outlines (Singapore's main island, its islands,
  and the Johor coast across the strait), the hydrographic features (canals, rivers, reservoirs, ponds, marshes), the
  parks and nature reserves, the runways and the CBD.
- silver.map_lines (25 Oct): the national map's lines: the contours (Contour_250K, every 20 m from -40 to 160 m, with
  the height in height_m), the expressways and their slip roads, the major roads, and the international boundary.

Each comes from the newest version data.gov.sg published (PUB's water-level sensors are in reference.py). Two layers
stop at bronze: SLA's vertical control points carry no heights (only each point's number, status and update time, and
every z is 0; checked again 25 Oct), so the heights come from the national map's contours; and the radar frames are
only archived for now.

A layer lands in bronze once per version, and silver keeps only the newest version (`_newest`), so a layer is a
snapshot, not a history. For land use, each Master Plan's table is read from its GeoJSON features: the properties
(URA's names contain dots, so they are quoted), the area URA measured, and the polygon kept as a GeoJSON string. The two
plans are stacked, each stamped with the days it is in force (MASTER_PLANS in utilities/silver_sources.py), so a
consumer picks the plan of a day by comparing it with valid_from and valid_to. The flood-prone area is a short yearly
table of hectares from the newest version of the CSV. Rows that don't parse, or lack a polygon or a positive area,
are dropped.

The national map's two tables read their features the same way: the layer (SLA's FOLDERPATH without its "Layers/"),
the name, the length or area SLA measured, and the geometry as a GeoJSON string (WGS84, each vertex with a z of 0). A
line that landed in parts (ingest/static.py feature_parts: a contour can hold 900,000 vertices) keeps a row per part,
each a LineString that starts where the part before it ends, numbered by `part` of `parts`; its length_m is the whole
line's. The map's inset of Pedra Branca, a frame and its labels, is map furniture, not ground, so it is left out (the
rock itself, at its true place 50 km east, stays among the coastal outlines). Features without a geometry (two
hydrographic ones) are dropped.
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


# The national map's inset of Pedra Branca: its frame and labels, drawn beside the rock, which are no part of the ground
MAP_FURNITURE = ("PB_draw_frame", "SMap_PedraBranca_Inset_Anno")


def _map_features(table: str) -> DataFrame:
    """The national map features of the newest version of a bronze table: layer, name, ids and the geometry."""
    rows = spark.read.table(table).select(
        get("version", "string").alias("version"),
        get("feature_index", "int").alias("feature_index"),
        # lines landed before parts existed are whole: part 0 of 1
        F.coalesce(get("part", "int"), F.lit(0)).alias("part"),
        F.coalesce(get("parts", "int"), F.lit(1)).alias("parts"),
        _property("OBJECTID", "bigint").alias("object_id"),
        F.regexp_replace(_property("FOLDERPATH", "string"), "^Layers/", "").alias(
            "layer"
        ),
        # SLA ends some names with a line break
        F.nullif(F.trim(_property("NAME", "string")), F.lit("")).alias("name"),
        _property("SHAPE.LEN", "double").alias("length_m"),
        _property("SHAPE.AREA", "double").alias("area_m2"),
        get("feature.geometry.type", "string").alias("geometry_type"),
        F.to_json(get("feature.geometry", "variant")).alias("geometry"),
    )
    # a row without a layer stays, for the layer_known expectation to count and drop
    return _newest(rows).where(
        F.coalesce(~F.col("layer").isin(*MAP_FURNITURE), F.lit(True))
    )


if landed(spark, "static_national_map_polygon"):

    @dp.materialized_view(
        name=f"{SILVER}.map_polygons",
        comment="The national map's polygons, one row per feature of the newest version: layer (Coastal_Outlines: "
        "Singapore's main island, its islands and the Johor coast; Hydrographic: canals, rivers, reservoirs, ponds "
        "and marshes; Parks_NaturalReserve; Airport_Runway; Central_Business_District), SLA's name and measured area "
        "(area_m2, in SVY21), and the polygon as GeoJSON (WGS84). The Pedra Branca inset's frame and labels are left "
        "out.",
    )
    @dp.expect_all_or_drop(
        {
            "layer_known": "layer IS NOT NULL",
            "polygon": "geometry_type IN ('Polygon', 'MultiPolygon') AND geometry IS NOT NULL",
        }
    )
    def map_polygons():
        return _map_features("static_national_map_polygon").select(
            "layer",
            "object_id",
            "name",
            "area_m2",
            "geometry_type",
            "geometry",
            "version",
        )


if landed(spark, "static_national_map_line"):

    @dp.materialized_view(
        name=f"{SILVER}.map_lines",
        comment="The national map's lines, one row per feature of the newest version, or per part of one that landed "
        "in parts (part of parts, consecutive LineStrings each starting where the one before ends): layer "
        "(Contour_250K: contours every 20 m, height_m their height, below 0 at sea; Expressway; Expressway_Sliproad; "
        "Major_Road; International_bdy), SLA's name and the whole line's measured length (length_m, in SVY21), and the "
        "line as GeoJSON (WGS84).",
    )
    @dp.expect_all_or_drop(
        {
            "layer_known": "layer IS NOT NULL",
            "line": "geometry_type IN ('LineString', 'MultiLineString') AND geometry IS NOT NULL",
        }
    )
    @dp.expect("contour_height_known", "layer <> 'Contour_250K' OR height_m IS NOT NULL")
    def map_lines():
        lines = _map_features("static_national_map_line")
        return lines.select(
            "layer",
            "object_id",
            "name",
            F.when(
                F.col("layer") == "Contour_250K", F.expr("try_cast(name AS INT)")
            ).alias("height_m"),
            "part",
            "parts",
            "length_m",
            "geometry_type",
            "geometry",
            "version",
        )
