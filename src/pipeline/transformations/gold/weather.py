"""Gold weather: what NEA's weather stations and lightning detection had shown at each 5-minute tick.

- gold.tick_weather: per tick, the island-wide weather (contract.py): the mean air temperature, humidity and wind over
  the stations, each station's newest reading in the WEATHER_MINUTES ending READING_LAG_MINUTES before the tick that NEA
  had published by the tick; how many stations each mean is over; and how many of NEA's 2-minute lightning records of
  the last LIGHTNING_MINUTES exist, which says whether the lightning feed was up.
- gold.area_lightning: per area and tick, the strikes within each of LIGHTNING_RADII_KM of the area's forecast point in
  the last LIGHTNING_MINUTES, as NEA had published them by the tick. A row only where a strike was in reach.

Wind is averaged as a vector (u eastward, v northward). NEA gives the direction the wind blows from, so the vector
points the other way. Every mean is of exact decimals, so a rebuild gives the same values however Spark splits the work.

How tick_weather is built (features.tick_weather, which flood-live's score task shares). As in gold.area_features'
rain, each reading is expanded into the ticks whose window holds it (the first tick at or after READING_LAG_MINUTES
past its timestamp, and the following ticks that WEATHER_MINUTES spans), and dropped from the ticks before NEA
published it. At each tick, a station's value for a variable is its
newest reading in the window; a mean over the stations then gives the island-wide value, and the number of stations
behind each mean is kept, so a mean over a few stations can be told from one over nearly all. Wind speed and direction
are combined into u and v components per station before averaging, because directions can't be averaged as numbers.
The lightning feed's health is a count of how many of the 2-minute records a tick's window holds; the table is a full
join of the weather and that count, so a tick with only lightning records still has a row.

How area_lightning is built. Counting every strike against every area at every tick would be a huge cross product, so
strikes are first binned to a 0.01-degree grid (GRID_DEG) with the ticks each counts at, from its publication until
the window has passed it. The cells within the largest radius of an area are found once, from the cell centres, and
the strikes are then summed per area and tick for each radius. That is why distances are off by up to 0.8 km, and why
a row exists only where a strike was within the largest radius.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.gold import (
    seconds,
    silver,
    tick_at_or_after,
    tick_before,
    ticks_between,
)
from utilities.silver import distance_km

from daisi_team_master_corgis_floodsense import features as F_
from daisi_team_master_corgis_floodsense.features import LIGHTNING_S
from daisi_team_master_corgis_floodsense.ml import contract as C

# Strikes are binned to a 0.01-degree grid (about 1.1 km) before their distance to each area is measured, so a
# distance is off by at most 0.8 km.
GRID_DEG = 0.01


@dp.materialized_view(
    name="tick_weather",
    comment="The island-wide weather at each 5-minute tick, as a live fetch at the tick would have had it: the mean "
    "air temperature (deg C), relative humidity (%) and wind (u eastward, v northward, knots) over NEA's weather "
    f"stations, each station's newest reading in the {C.WEATHER_MINUTES} minutes ending "
    f"{C.READING_LAG_MINUTES} min before the tick that NEA had published by it; the stations each mean is over; and "
    f"lightning_records, how many of NEA's {C.LIGHTNING_RECORDS_PER_WINDOW} two-minute lightning records of the last "
    f"{C.LIGHTNING_MINUTES} minutes exist (fewer means the lightning feed was down). The same for every area.",
    cluster_by=["ts"],
)
@dp.expect_all(
    {
        "temperature_plausible": "air_temp_c IS NULL OR air_temp_c BETWEEN 15 AND 40",
        "humidity_plausible": "rel_humidity_pct IS NULL OR rel_humidity_pct BETWEEN 0 AND 100",
        "wind_plausible": "wind_u_kn IS NULL OR sqrt(wind_u_kn * wind_u_kn + wind_v_kn * wind_v_kn) <= 60",
        "records_in_window": f"lightning_records BETWEEN 0 AND {C.LIGHTNING_RECORDS_PER_WINDOW}",
    }
)
def tick_weather():
    return F_.tick_weather(
        spark.read.table(silver(spark, "weather_1min")),
        spark.read.table(silver(spark, "lightning_records")),
    )


@dp.materialized_view(
    name="area_lightning",
    comment="Lightning strikes NEA observed near each forecast area's label point in the "
    f"{C.LIGHTNING_MINUTES} minutes before each tick, as NEA had published them by the tick: one column per radius "
    f"({', '.join(f'{km} km' for km in C.LIGHTNING_RADII_KM)}). A row only where a strike was within "
    f"{max(C.LIGHTNING_RADII_KM)} km; gold.area_features reads a missing row as none where the feed was up "
    "(gold.tick_weather.lightning_records). Distances are from a 0.01-degree grid, so off by at most 0.8 km.",
    cluster_by=["ts", "area"],
)
@dp.expect(
    "radii_nested",
    f"{C.lightning_column(min(C.LIGHTNING_RADII_KM))} <= "
    f"{C.lightning_column(max(C.LIGHTNING_RADII_KM))}",
)
def area_lightning():
    strikes = spark.read.table(silver(spark, "lightning_strikes")).where(
        "ts IS NOT NULL AND published_at IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL"
    )
    # a strike counts at the ticks whose window (t - LIGHTNING_MINUTES, t] holds it, from its publication on
    strike_s = F.unix_micros("ts") / 1_000_000
    binned = (
        strikes.select(
            F.floor(F.col("lat") / GRID_DEG).alias("cell_lat"),
            F.floor(F.col("lon") / GRID_DEG).alias("cell_lon"),
            tick_at_or_after(seconds("published_at")).alias("first_s"),
            tick_before(strike_s + LIGHTNING_S).alias("last_s"),
        )
        .groupBy("cell_lat", "cell_lon", "first_s", "last_s")
        .agg(F.count("*").alias("n"))
    )
    per_cell = (
        binned.select(
            "cell_lat",
            "cell_lon",
            "n",
            ticks_between(F.col("first_s"), F.col("last_s")).alias("tick_s"),
        )
        .groupBy("cell_lat", "cell_lon", "tick_s")
        .agg(F.sum("n").alias("strikes"))
    )
    areas = spark.read.table(silver(spark, "forecast_areas")).select(
        "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
    )
    km = distance_km(
        F.col("area_lat"),
        F.col("area_lon"),
        (F.col("cell_lat") + 0.5) * GRID_DEG,
        (F.col("cell_lon") + 0.5) * GRID_DEG,
    )
    reach = (
        per_cell.select("cell_lat", "cell_lon")
        .distinct()
        .crossJoin(F.broadcast(areas))
        .withColumn("km", km)
        .where(F.col("km") <= max(C.LIGHTNING_RADII_KM))
        .select("cell_lat", "cell_lon", "area", "km")
    )
    near = per_cell.join(reach, ["cell_lat", "cell_lon"])
    counts = near.groupBy("area", "tick_s").agg(
        *(
            F.coalesce(F.sum(F.when(F.col("km") <= r, F.col("strikes"))), F.lit(0))
            .cast("int")
            .alias(C.lightning_column(r))
            for r in C.LIGHTNING_RADII_KM
        )
    )
    return counts.select(
        "area",
        F.timestamp_seconds("tick_s").alias("ts"),
        *(C.lightning_column(r) for r in C.LIGHTNING_RADII_KM),
    )
