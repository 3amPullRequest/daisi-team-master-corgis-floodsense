"""gold.area_terrain: the ground of each NEA forecast area, from the national map (silver.map_lines, silver.map_polygons):
how far its land lies from the hills, from the sea and from a canal or river, how high the hills around it rise, and
how much of it trunk roads and parks take up (contract.MAP_FEATURES). One row per area, the same at every tick:
gold.area_features carries the columns to the models that ask for them (add=map_line, add=map_polygon), and
flood-live's score task reads this table as gold does.

An area's land is where its alerts are placed: the land nearer its label point than any other area's (its Voronoi
cell), as in gold.area_land_use. The land is Singapore's coastal outlines, the main island and its islands (the Johor
coast across the strait is left out, and so is Pedra Branca, 50 km east and in no forecast area), filled with H3 cells
of resolution 11 (about 0.0025 km^2 each); each cell belongs to the area whose label point is nearest its centre
(utilities/gold.nearest_area).

The columns, each over the area's land cells:

- high_ground_km: the mean distance to the nearest contour of HIGH_GROUND_M or more over Singapore's land. The map's
  contours are every 20 m, so this is how far the hills are, whose rain runs down onto the land below them.
- relief_m: the highest such contour within GAUGE_RADIUS_KM of the label point (the reach of the area's rain gauges),
  0 where there is none: how high those hills rise.
- road_share: the share of the land cells that an expressway, slip road or major road crosses (their centre lines,
  covered with resolution-11 cells): how much of the land is trunk road.
- coast_km: the mean distance to the sea, that is to the coast cells: land cells with a neighbour that isn't land
  (the sea, or a basin enclosed by reclaimed land). Where the land is near the sea, a high tide slows the drains.
- waterway_km: the mean distance to the nearest canal or river: the hydrographic polygons named as one (WATERWAY_WORDS;
  reservoirs, ponds, lakes, quarries and marshes aren't).
- green_share: the share of the land cells in parks and nature reserves, where rain soaks in rather than runs off.

How distances are measured: between the centres of resolution-9 cells (about 0.1 km^2, 174 m edges). Each land cell
reads the distance of its resolution-9 parent, and a target (a contour, a coast cell, a canal) is every resolution-9
cell it covers that holds Singapore's land, so the Johor hills across the strait don't count. A distance is thus good to
about 0.2 km. With about 7,500 parent cells and a few thousand target cells, the nearest is found by brute force: every
pair, the shortest kept. Distances are averaged as exact decimals and shares are ratios of counts, so a rebuild gives the
same values however Spark splits the work.

The map is a snapshot (silver keeps its newest version), read at every tick since 2021: canals, parks and the reclaimed
coast have changed little in the five years, but not nothing.
"""

from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from utilities.gold import cell_point, nearest_area, silver
from utilities.silver import distance_km, in_singapore

from daisi_team_master_corgis_floodsense.ml import contract as C

LAND_RESOLUTION = 11  # the land, its parks and roads, as gold.area_land_use
DISTANCE_RESOLUTION = 9  # what distances are measured between
# the map's line layers that are trunk roads, and the words that make a hydrographic polygon a canal or river (sungei is
# Malay for river)
ROAD_LAYERS = ("Expressway", "Expressway_Sliproad", "Major_Road")
WATERWAY_WORDS = ("CANAL", "RIVER", "SUNGEI", "CREEK", "WATERWAY", "DRAIN")
KM = "decimal(9,6)"  # a distance to the millimetre, averaged exactly


def _cells(shapes: DataFrame, resolution: int, cover: bool = False) -> DataFrame:
    """The distinct H3 cells of the shapes' geometries: those whose centre lies inside a polygon (polyfill), or, with
    `cover`, every cell a line or polygon touches."""
    fill = "h3_coverash3" if cover else "h3_polyfillash3"
    return shapes.select(
        F.explode(F.expr(f"{fill}(geometry, {resolution})")).alias("cell")
    ).distinct()


def _nearest_km(parents: DataFrame, targets: DataFrame, name: str) -> DataFrame:
    """(parent, `name`): per cell of `parents`, the distance (km) from its centre to the nearest centre of a cell of
    `targets`."""
    lat, lon = cell_point("parent")
    t_lat, t_lon = cell_point("cell")
    km = distance_km(F.col("lat"), F.col("lon"), F.col("t_lat"), F.col("t_lon"))
    return (
        parents.select("parent", lat.alias("lat"), lon.alias("lon"))
        .crossJoin(F.broadcast(targets.select(t_lat.alias("t_lat"), t_lon.alias("t_lon"))))
        .groupBy("parent")
        .agg(F.min(km).cast(KM).alias(name))
    )


@dp.materialized_view(
    name=C.AREA_TERRAIN,
    comment="The ground of each NEA forecast area from the national map, one row per area, the same at every tick: "
    f"over the area's land (its Voronoi cell, filled with H3 cells of resolution {LAND_RESOLUTION}), the mean "
    f"distance to a contour of {C.HIGH_GROUND_M} m or more (high_ground_km), to the sea (coast_km) and to a canal or "
    f"river (waterway_km), measured between resolution-{DISTANCE_RESOLUTION} cells; the highest contour within "
    f"{C.GAUGE_RADIUS_KM:g} km of the label point (relief_m); and the share of the land cells crossed by a trunk road "
    "(road_share) or in a park or nature reserve (green_share). gold.area_features carries them for the models.",
)
@dp.expect_all(
    {
        "every_area_has_land": "land_cells > 0",
        "distances_known": "high_ground_km >= 0 AND coast_km >= 0 AND waterway_km >= 0",
        "shares_in_range": "road_share BETWEEN 0 AND 1 AND green_share BETWEEN 0 AND 1",
        # Bukit Timah Hill, the island's highest, is 163 m
        "relief_plausible": "relief_m BETWEEN 0 AND 180",
    }
)
def area_terrain():
    polygons = spark.read.table(silver(spark, "map_polygons"))
    lines = spark.read.table(silver(spark, "map_lines"))
    areas = spark.read.table(silver(spark, "forecast_areas"))

    singapore = (F.col("layer") == "Coastal_Outlines") & ~F.coalesce(
        F.col("name"), F.lit("")
    ).contains("MALAYSIA")
    lat, lon = cell_point()
    land = (
        _cells(polygons.where(singapore), LAND_RESOLUTION)
        .select("cell", lat.alias("lat"), lon.alias("lon"))
        .where(in_singapore())
        .select("cell")
    )
    placed = nearest_area(land, areas).withColumn(
        "parent", F.expr(f"h3_toparent(cell, {DISTANCE_RESOLUTION})")
    )
    parents = placed.select("parent").distinct()
    on_land = parents.select(F.col("parent").alias("cell"))

    # the targets, as resolution-9 cells over Singapore's land
    high = (
        lines.where(
            (F.col("layer") == "Contour_250K") & (F.col("height_m") >= C.HIGH_GROUND_M)
        )
        .select(
            F.explode(
                F.expr(f"h3_coverash3(geometry, {DISTANCE_RESOLUTION})")
            ).alias("cell"),
            "height_m",
        )
        .groupBy("cell")
        .agg(F.max("height_m").alias("height_m"))
        .join(on_land, "cell")
    )
    neighbours = land.select(
        "cell", F.explode(F.expr("h3_kring(cell, 1)")).alias("neighbour")
    )
    coast = (
        neighbours.join(
            land.select(F.col("cell").alias("neighbour")), "neighbour", "left_anti"
        )
        .select(F.expr(f"h3_toparent(cell, {DISTANCE_RESOLUTION})").alias("cell"))
        .distinct()
    )
    waterway = (F.col("layer") == "Hydrographic") & F.col("name").rlike(
        "|".join(WATERWAY_WORDS)
    )
    waterways = _cells(
        polygons.where(waterway), DISTANCE_RESOLUTION, cover=True
    ).join(on_land, "cell")
    distances = (
        _nearest_km(parents, high, "high_ground_km")
        .join(_nearest_km(parents, coast, "coast_km"), "parent")
        .join(_nearest_km(parents, waterways, "waterway_km"), "parent")
    )

    # the shares, of resolution-11 land cells
    parks = _cells(
        polygons.where(F.col("layer") == "Parks_NaturalReserve"), LAND_RESOLUTION
    ).select("cell", F.lit(1).alias("green"))
    roads = _cells(
        lines.where(F.col("layer").isin(*ROAD_LAYERS)), LAND_RESOLUTION, cover=True
    ).select("cell", F.lit(1).alias("road"))
    land_cells = F.count("*")
    per_area = (
        placed.join(distances, "parent")
        .join(parks, "cell", "left")
        .join(roads, "cell", "left")
        .groupBy("area")
        .agg(
            land_cells.cast("int").alias("land_cells"),
            *(
                F.avg(c).cast("double").alias(c)
                for c in ("high_ground_km", "coast_km", "waterway_km")
            ),
            (F.count("road") / land_cells).alias("road_share"),
            (F.count("green") / land_cells).alias("green_share"),
        )
    )

    # the hills within reach of the label point
    h_lat, h_lon = cell_point("cell")
    km = distance_km(F.col("lat"), F.col("lon"), F.col("h_lat"), F.col("h_lon"))
    relief = (
        areas.select("area", "lat", "lon")
        .crossJoin(
            F.broadcast(
                high.select(h_lat.alias("h_lat"), h_lon.alias("h_lon"), "height_m")
            )
        )
        .where(km <= C.GAUGE_RADIUS_KM)
        .groupBy("area")
        .agg(F.max("height_m").alias("relief_m"))
    )
    return (
        areas.select("area")
        .join(per_area, "area", "left")
        .join(relief, "area", "left")
        .select(
            "area",
            "land_cells",
            "high_ground_km",
            F.coalesce(F.col("relief_m"), F.lit(0)).cast("double").alias("relief_m"),
            "road_share",
            "coast_km",
            "waterway_km",
            "green_share",
        )
    )
