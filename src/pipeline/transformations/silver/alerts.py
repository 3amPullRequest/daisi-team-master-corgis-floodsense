"""Silver alerts: every PUB alert placed in an NEA forecast area.

- silver.location_gazetteer: one row per alert location (normalised text) with its coordinates, from the first of: a
  hand fix (data/static/location_gazetteer_manual.csv), PUB's own circle (flood-alerts API), the water-level sensor of
  the same name (the 90% warnings come from those sensors), a OneMap search (bronze.onemap_search). Its area is the
  forecast area whose label point is nearest, which is the same as falling in that point's Voronoi cell.
- silver.alerts: every 90% warning and flash flood, from Telegram and the API, one row per location a post names,
  with its place. An API flood that Telegram also posted is the same alert, marked as such.
- silver.area_alerts: the labels the models read, exactly ml/contract.py AREA_ALERTS_COLUMNS: placed alerts, each once.
- silver.alerts_unplaced: alerts no method could place, to fix by hand.

The steps, in the order the code runs them (a step is built only when its sources have landed):

1. `alert_locations` (a temporary view) puts both sources into one table of alert locations. A Telegram post that names
   several places gives a row for each, timed by that place's stamp, else by the posting. An API alert gives one row,
   with PUB's own circle. Every row gets a `location_key`, so spellings of one place are one place.
2. `api_alert_matches` finds the API floods that are also Telegram posts, because PUB wrote those twice and an alert
   must count once. A pair has the same kind, is stamped within SAME_ALERT_S of each other, and names the place alike
   (a looser name score than the gazetteer's, since PUB words the API's text apart from the post's); the closest in
   time wins.
3. `location_gazetteer` places each location, taking the first method that works: a hand fix; PUB's circle (also given
   to the Telegram location that is the same alert); the water-level sensor best named like it, accepted only if it
   scores at least SAME_NAME and no other sensor names it as well from more than SAME_PLACE_KM away; or the first
   OneMap query whose top result is in Singapore and names the place. The area is then the nearest of the 47 label
   points. The table also keeps the best sensor and OneMap candidates, and how far OneMap's answer lies from the chosen
   point, so a wrong placement can be spotted.
4. `alerts` is every alert row with its place and area, an API flood that is a Telegram post's alert marked with that
   post's id (`same_as_message_id`).
5. `area_alerts` is what the models and the scoreboard read: the placed alerts, API duplicates left out so that
   Telegram's stamp is kept, one row per area, kind, time and location. An expectation drops any row without an area.
6. `alerts_unplaced` is the quarantine: the alerts no method placed, each with the best sensor and OneMap candidates as
   a start for a hand fix. gold.label_ticks reads it, so that an alert that could belong to any area hides the labels
   around it, rather than quietly missing from some area's.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.contract_tables import AREA_ALERTS_SCHEMA
from utilities.silver import (
    distance_km,
    get,
    in_singapore,
    landed,
    location_key,
    name_match,
    schema,
    seed,
)

from daisi_team_master_corgis_floodsense.labels.locations import SAME_NAME
from daisi_team_master_corgis_floodsense.ml import contract as C

SILVER = schema(spark)
TELEGRAM = bool(landed(spark, "pub_telegram_posts"))
API = bool(landed(spark, "pub_flood_alerts"))
AREAS = bool(landed(spark, "nea_two_hr_forecast", "nea_two_hr_forecast_hist"))
SENSORS = bool(landed(spark, "static_pub_water_level_sensors"))
ONEMAP = bool(landed(spark, "onemap_search"))

# Two sensors this close are one place, so a tie between them leaves no doubt.
SAME_PLACE_KM = 1.0
# An API flood is a Telegram post's alert when it is stamped within SAME_ALERT_S of it and names the place alike, from
# SAME_ALERT_NAME: PUB writes the API's text apart from the post's ('Boon Lay Avenue' for 'Boon Lay Ave (Between Jalan
# Boon Lay and Boon Lay Drive)'), and two floods can come minutes apart in one area (12 Dec 2025, Bedok).
SAME_ALERT_S = 15 * 60
SAME_ALERT_NAME = 0.5
NULL_DOUBLE = F.lit(None).cast("double")
NULL_STRING = F.lit(None).cast("string")


def _sensor_matches(keys):
    """Per location: the best-named sensor, and whether it is clear enough to place the location by."""
    sensors = spark.read.table(f"{SILVER}.water_level_sensors").select(
        "sensor_id",
        F.col("name").alias("sensor_name"),
        F.col("lat").alias("s_lat"),
        F.col("lon").alias("s_lon"),
    )
    scored = (
        keys.select("location_key", "location")
        .crossJoin(F.broadcast(sensors))
        .withColumn("score", name_match("location", "sensor_name"))
        .where("score > 0")
    )
    rank = Window.partitionBy("location_key").orderBy(F.desc("score"), "sensor_id")
    ranked = scored.withColumn("rank", F.row_number().over(rank)).where("rank <= 2")
    second = ranked.where("rank = 2").select(
        "location_key",
        F.col("score").alias("score2"),
        F.col("s_lat").alias("lat2"),
        F.col("s_lon").alias("lon2"),
    )
    pair = ranked.where("rank = 1").join(second, "location_key", "left")
    clear = (
        F.col("score2").isNull()
        | (F.col("score2") < F.col("score"))
        | (
            distance_km(F.col("s_lat"), F.col("s_lon"), F.col("lat2"), F.col("lon2"))
            <= SAME_PLACE_KM
        )
    )
    accepted = (F.col("score") >= SAME_NAME) & clear
    return pair.select(
        "location_key",
        F.concat_ws(" ", "sensor_id", "sensor_name").alias("sensor_candidate"),
        F.col("score").alias("sensor_candidate_score"),
        F.when(accepted, F.col("s_lat")).alias("sensor_lat"),
        F.when(accepted, F.col("s_lon")).alias("sensor_lon"),
    )


def _onemap_matches():
    """Per location: the first of its OneMap queries (best guess first) whose top result is in Singapore and names the
    place searched, as ingest/onemap.py `accepted` decides ("Kings Rd" -> Kings Home on Joo Chiat Road is refused)."""

    def top(field: str):
        return get(f"body.results[0].{field}", "string").alias(field.lower())

    rows = spark.read.table("onemap_search").select(
        get("location", "string").alias("location"),
        get("query", "string").alias("query"),
        get("query_rank", "int").alias("query_rank"),
        F.expr(
            "try_cast(try_variant_get(payload, '$.body.results[0].LATITUDE', 'string') AS DOUBLE)"
        ).alias("lat"),
        F.expr(
            "try_cast(try_variant_get(payload, '$.body.results[0].LONGITUDE', 'string') AS DOUBLE)"
        ).alias("lon"),
        top("ADDRESS"),
        top("ROAD_NAME"),
        top("SEARCHVAL"),
        top("BUILDING"),
        "fetched_at",
    )
    score = F.greatest(
        *(
            name_match("query", F.coalesce(c, F.lit("")))
            for c in ("road_name", "searchval", "building")
        )
    )
    found = (
        rows.where(in_singapore())
        .withColumn("onemap_score", score)
        .where(F.col("onemap_score") >= SAME_NAME)
        .withColumn("location_key", location_key("location"))
    )
    best = Window.partitionBy("location_key").orderBy(
        "query_rank", F.desc("fetched_at")
    )
    return (
        found.withColumn("_n", F.row_number().over(best))
        .where("_n = 1")
        .select(
            "location_key",
            F.col("lat").alias("onemap_lat"),
            F.col("lon").alias("onemap_lon"),
            F.concat_ws(" -> ", "query", "address").alias("onemap_match"),
            F.col("query_rank").alias("onemap_query_rank"),
        )
    )


if (TELEGRAM or API) and AREAS:

    @dp.temporary_view(name="alert_locations")
    def alert_locations():
        """One row per alert location: Telegram posts name one or more, an API alert one (with PUB's coordinates)."""
        parts = []
        if TELEGRAM:
            posts = spark.read.table(f"{SILVER}.telegram_posts").where(
                F.col("category").isin(*C.ALERT_KINDS)
            )
            named = posts.select(
                "message_id",
                "link",
                F.col("category").alias("kind"),
                "posted_at",
                F.explode("locations").alias("place"),
            )
            parts.append(
                named.select(
                    F.lit("telegram").alias("source"),
                    "message_id",
                    "link",
                    "kind",
                    F.col("place.location").alias("location"),
                    F.coalesce("place.stamp_ts", "posted_at").alias("alert_ts"),
                    F.col("posted_at").alias("posted_ts"),
                    F.when(F.col("place.stamp_ts").isNull(), "posting")
                    .otherwise("stamp")
                    .alias("timed_by"),
                    NULL_DOUBLE.alias("pub_lat"),
                    NULL_DOUBLE.alias("pub_lon"),
                )
            )
        if API:
            api = spark.read.table(f"{SILVER}.api_flood_alerts").where(
                F.col("category").isin(*C.ALERT_KINDS)
            )
            parts.append(
                api.select(
                    F.lit("api").alias("source"),
                    F.lit(None).cast("bigint").alias("message_id"),
                    NULL_STRING.alias("link"),
                    F.col("category").alias("kind"),
                    "location",
                    "alert_ts",
                    F.col("issued_at").alias("posted_ts"),
                    F.when(F.col("stamp").isNull(), "posting")
                    .otherwise("stamp")
                    .alias("timed_by"),
                    F.col("lat").alias("pub_lat"),
                    F.col("lon").alias("pub_lon"),
                )
            )
        rows = parts[0] if len(parts) == 1 else parts[0].unionByName(parts[1])
        return rows.withColumn("location_key", location_key("location"))

    @dp.temporary_view(name="api_alert_matches")
    def api_alert_matches():
        """Each API alert that is also a Telegram post's: same kind, stamped within SAME_ALERT_S, and named alike; the
        closest in time when several are. Empty without API alerts."""
        rows = spark.read.table("alert_locations")
        api = rows.where("source = 'api'").select(
            "alert_ts", "location_key", "kind", "location", "pub_lat", "pub_lon"
        )
        telegram = rows.where("source = 'telegram'").select(
            F.col("message_id").alias("t_message_id"),
            F.col("kind").alias("t_kind"),
            F.col("alert_ts").alias("t_ts"),
            F.col("location").alias("t_location"),
            F.col("location_key").alias("t_location_key"),
        )
        apart = F.abs(api["alert_ts"].cast("long") - telegram["t_ts"].cast("long"))
        pairs = (
            api.join(
                telegram,
                (api["kind"] == telegram["t_kind"]) & (apart <= SAME_ALERT_S),
            )
            .withColumn("apart_s", apart)
            .withColumn("similarity", name_match("location", "t_location"))
            .where(F.col("similarity") >= SAME_ALERT_NAME)
        )
        closest = Window.partitionBy("alert_ts", "location_key").orderBy(
            "apart_s", F.desc("similarity"), "t_message_id"
        )
        return (
            pairs.withColumn("_n", F.row_number().over(closest))
            .where("_n = 1")
            .select(
                "alert_ts",
                "location_key",
                "t_message_id",
                "t_location_key",
                "pub_lat",
                "pub_lon",
                "similarity",
            )
        )

    @dp.materialized_view(
        name=f"{SILVER}.location_gazetteer",
        comment="Every alert location (location_key: the normalised text) with its coordinates and forecast area. "
        "method is the first that placed it: 'manual' (data/static/location_gazetteer_manual.csv), 'pub_api' (PUB's "
        "circle), 'pub_sensor' (the water-level sensor of the same name), 'onemap' (OneMap search). onemap_km is how "
        "far OneMap's answer lies from the chosen point, a cross-check; area_km is the distance to the area's point.",
    )
    @dp.expect("placed", "area IS NOT NULL")
    def location_gazetteer():
        alerts = spark.read.table("alert_locations").where("location_key <> ''")
        spelling = (
            alerts.groupBy("location_key", "location")
            .count()
            .groupBy("location_key")
            .agg(F.max_by("location", F.struct("count", "location")).alias("location"))
        )
        keys = (
            alerts.groupBy("location_key")
            .agg(
                F.count("*").alias("alerts"),
                F.array_sort(F.collect_set("kind")).alias("kinds"),
                F.min("alert_ts").alias("first_alert"),
                F.max("alert_ts").alias("last_alert"),
            )
            .join(spelling, "location_key")
        )

        manual = (
            seed(
                spark,
                "location_gazetteer_manual",
                {
                    "location": "STRING",
                    "lat": "DOUBLE",
                    "lon": "DOUBLE",
                    "note": "STRING",
                },
            )
            .where("location IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL")
            .select(
                location_key("location").alias("location_key"),
                F.col("lat").alias("manual_lat"),
                F.col("lon").alias("manual_lon"),
                F.col("note").alias("manual_note"),
            )
            .dropDuplicates(["location_key"])
        )
        # PUB's circle places the API alert's location, and the Telegram post's location that is the same alert
        circles = alerts.select("location_key", "pub_lat", "pub_lon").unionByName(
            spark.read.table("api_alert_matches").select(
                F.col("t_location_key").alias("location_key"), "pub_lat", "pub_lon"
            )
        )
        pub = (
            circles.where("pub_lat IS NOT NULL AND pub_lon IS NOT NULL")
            .groupBy("location_key")
            .agg(F.avg("pub_lat").alias("pub_lat"), F.avg("pub_lon").alias("pub_lon"))
        )
        located = keys.join(manual, "location_key", "left").join(
            pub, "location_key", "left"
        )
        if SENSORS:
            located = located.join(_sensor_matches(keys), "location_key", "left")
        else:
            located = located.select(
                "*",
                NULL_STRING.alias("sensor_candidate"),
                NULL_DOUBLE.alias("sensor_candidate_score"),
                NULL_DOUBLE.alias("sensor_lat"),
                NULL_DOUBLE.alias("sensor_lon"),
            )
        if ONEMAP:
            located = located.join(_onemap_matches(), "location_key", "left")
        else:
            located = located.select(
                "*",
                NULL_DOUBLE.alias("onemap_lat"),
                NULL_DOUBLE.alias("onemap_lon"),
                NULL_STRING.alias("onemap_match"),
                F.lit(None).cast("int").alias("onemap_query_rank"),
            )

        method = (
            F.when(F.col("manual_lat").isNotNull(), "manual")
            .when(F.col("pub_lat").isNotNull(), "pub_api")
            .when(F.col("sensor_lat").isNotNull(), "pub_sensor")
            .when(F.col("onemap_lat").isNotNull(), "onemap")
        )
        matched = (
            F.when(
                F.col("manual_lat").isNotNull(),
                F.coalesce("manual_note", F.lit("hand fix")),
            )
            .when(F.col("pub_lat").isNotNull(), F.lit("PUB flood-alerts API circle"))
            .when(F.col("sensor_lat").isNotNull(), F.col("sensor_candidate"))
            .when(F.col("onemap_lat").isNotNull(), F.col("onemap_match"))
        )
        chosen = located.select(
            "location_key",
            "location",
            "alerts",
            "kinds",
            "first_alert",
            "last_alert",
            method.alias("method"),
            F.coalesce("manual_lat", "pub_lat", "sensor_lat", "onemap_lat").alias(
                "lat"
            ),
            F.coalesce("manual_lon", "pub_lon", "sensor_lon", "onemap_lon").alias(
                "lon"
            ),
            matched.alias("matched"),
            "sensor_candidate",
            "sensor_candidate_score",
            "onemap_lat",
            "onemap_lon",
            "onemap_match",
            "onemap_query_rank",
        ).withColumn(
            "onemap_km",
            F.when(
                F.col("method") != "onemap",
                distance_km(
                    F.col("lat"), F.col("lon"), F.col("onemap_lat"), F.col("onemap_lon")
                ),
            ),
        )

        areas = spark.read.table(f"{SILVER}.forecast_areas").select(
            "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
        )
        nearest = (
            chosen.where("lat IS NOT NULL AND lon IS NOT NULL")
            .select("location_key", "lat", "lon")
            .crossJoin(F.broadcast(areas))
            .withColumn(
                "km",
                distance_km(
                    F.col("lat"), F.col("lon"), F.col("area_lat"), F.col("area_lon")
                ),
            )
            .groupBy("location_key")
            .agg(F.min_by("area", "km").alias("area"), F.min("km").alias("area_km"))
        )
        return chosen.join(nearest, "location_key", "left")

    @dp.materialized_view(
        name=f"{SILVER}.alerts",
        comment="Every PUB alert (90% warnings and flash floods, from Telegram and the flood-alerts API), one row per "
        "location a post names, with its place and forecast area (from silver.location_gazetteer). alert_ts is PUB's "
        "stamp, or the posting when a post has none (timed_by). same_as_message_id marks an API flood that is a "
        "Telegram post's alert (same kind, stamped within 15 minutes, named alike).",
    )
    def alerts():
        rows = spark.read.table("alert_locations")
        places = spark.read.table(f"{SILVER}.location_gazetteer").select(
            "location_key",
            "area",
            "lat",
            "lon",
            F.col("method").alias("geocoded_by"),
            "area_km",
        )
        located = rows.join(places, "location_key", "left")
        same = (
            spark.read.table("api_alert_matches")
            .select(
                "alert_ts",
                "location_key",
                F.col("t_message_id").alias("same_as_message_id"),
            )
            .withColumn("source", F.lit("api"))
        )
        return located.join(
            same, ["source", "alert_ts", "location_key"], "left"
        ).select(
            "source",
            "message_id",
            "link",
            "kind",
            "location",
            "location_key",
            "alert_ts",
            "posted_ts",
            "timed_by",
            "area",
            "lat",
            "lon",
            "geocoded_by",
            "area_km",
            "same_as_message_id",
        )

    @dp.materialized_view(
        name=f"{SILVER}.area_alerts",
        comment="PUB alerts by NEA forecast area: the labels every FloodSense model and the scoreboard read "
        "(ml/contract.py AREA_ALERTS_COLUMNS). Placed alerts only, each once; an alert seen on Telegram and the API "
        "keeps Telegram's stamp. Not alerts: flash flood subsided, and NEA heavy-rain posts.",
        schema=AREA_ALERTS_SCHEMA,
        cluster_by=["alert_ts"],
    )
    @dp.expect_or_drop("placed", "area IS NOT NULL")
    @dp.expect("known_kind", f"kind IN {tuple(C.ALERT_KINDS)}")
    def area_alerts():
        alerts = spark.read.table(f"{SILVER}.alerts").where(
            "same_as_message_id IS NULL"
        )
        return alerts.select(*C.AREA_ALERTS_COLUMNS).dropDuplicates(
            ["area", "kind", "alert_ts", "location"]
        )

    @dp.materialized_view(
        name=f"{SILVER}.alerts_unplaced",
        comment="Alerts no method could place in an area (the quarantine): add each location with its lat/lon to "
        "data/static/location_gazetteer_manual.csv. sensor_candidate is the best-named sensor, for a start.",
    )
    def alerts_unplaced():
        hints = spark.read.table(f"{SILVER}.location_gazetteer").select(
            "location_key", "sensor_candidate", "sensor_candidate_score", "onemap_match"
        )
        return (
            spark.read.table(f"{SILVER}.alerts")
            .where("area IS NULL")
            .join(hints, "location_key", "left")
            .select(
                "source",
                "message_id",
                "link",
                "kind",
                "location",
                "location_key",
                "alert_ts",
                "sensor_candidate",
                "sensor_candidate_score",
                "onemap_match",
            )
        )
