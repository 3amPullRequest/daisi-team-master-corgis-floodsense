"""Gold context: what each area is made of, and the trend in PUB's flood-prone land, for the app and the dashboard.

- gold.area_land_use: per area and Master Plan, the share of the area in each land use. Each polygon of
  silver.land_use is filled with H3 cells (resolution 11, about 0.0025 km^2 each here), and a cell belongs to the area
  whose label point is nearest, as an alert does (the area's Voronoi cell). So a share is of the planned land and water
  in the area, and a polygon that spans two areas counts in both, cell by cell.
- gold.flood_prone_trend: PUB's flood-prone hectares by year, with the change from the year before.

Neither is a model input: a share is the same at every tick of an area, and the models already read the area. (The
ground from the national map, gold.area_terrain in terrain.py, is the same at every tick too, and gold.area_features
carries it for the models that ask for it, to test whether it changes what the weather does.)

How area_land_use is built. Land use is a set of polygons, but an area is a Voronoi cell around a label point, which
the data doesn't contain as polygons. So the polygons are turned into points: each is filled with H3 cells, and each
distinct cell is assigned to the area whose label point is nearest its centre (ties go to the first area by name, so a
rebuild assigns every cell the same way). Counting a plan's cells per area and land use, and dividing by the area's
cells in that plan, gives the share. A polygon spanning two areas thus counts in both, cell by cell. Plans are kept
apart, with the days each is in force. flood_prone_trend is a plain year-on-year difference of PUB's hectares.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.gold import nearest_area, silver

H3_RESOLUTION = 11


@dp.materialized_view(
    name="area_land_use",
    comment="URA Master Plan land use in each NEA forecast area: per area, plan and land use, the H3 cells "
    f"(resolution {H3_RESOLUTION}, about 0.0025 km^2) of that use whose centre is nearer the area's label point than "
    "any other's, and their share of the area's cells. valid_from and valid_to are the days the plan is in force, as "
    "in silver.land_use. Planned use, not land cover.",
)
@dp.expect_all(
    {
        "share_in_range": "share > 0 AND share <= 1",
        "plan_known": "plan IN ('MP2019', 'MP2025')",
    }
)
def area_land_use():
    land = spark.read.table(silver(spark, "land_use")).select(
        "plan",
        "valid_from",
        "valid_to",
        "land_use",
        F.explode(F.expr(f"h3_polyfillash3(geometry, {H3_RESOLUTION})")).alias("cell"),
    )
    # ties go to the first area by name, so a rebuild places every cell the same way
    nearest = nearest_area(land, spark.read.table(silver(spark, "forecast_areas")))
    counted = (
        land.join(nearest, "cell")
        .groupBy("area", "plan", "valid_from", "valid_to", "land_use")
        .agg(F.count("*").alias("cells"))
    )
    in_area = Window.partitionBy("area", "plan")
    return counted.select(
        "area",
        "plan",
        "valid_from",
        "valid_to",
        "land_use",
        F.col("cells").cast("int").alias("cells"),
        (F.col("cells") / F.sum("cells").over(in_area)).alias("share"),
    )


@dp.materialized_view(
    name="flood_prone_trend",
    comment="PUB's total flood-prone area by year (silver.flood_prone_area), with the change in hectares and percent "
    "from the year before: the dashboard's trend in flood-prone land.",
)
def flood_prone_trend():
    years = spark.read.table(silver(spark, "flood_prone_area")).select(
        "year", "hectares"
    )
    before = F.lag("hectares").over(Window.orderBy("year"))
    # rounded as PUB reports the hectares, to one decimal
    return years.select(
        "year",
        "hectares",
        F.round(F.col("hectares") - before, 1).alias("change_ha"),
        F.round((F.col("hectares") - before) / before * 100, 1).alias("change_pct"),
    )
