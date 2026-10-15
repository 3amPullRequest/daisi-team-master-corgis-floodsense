"""silver.api_flood_alerts: PUB's flood-alerts API (CAP), one row per alert, read like a Telegram post.

The API carries only flash floods and their "subsided" notices (no 90% warnings), and its `description` is the text of
PUB's Telegram post, so labels/alert_text.py classifies it the same way. Unlike Telegram, it gives PUB's own
coordinates for the location (a circle), which silver.location_gazetteer uses first.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.silver import get, get_ts, in_singapore, landed, parse_alert, schema

SILVER = schema(spark)

if landed(spark, "pub_flood_alerts"):

    @dp.materialized_view(
        name=f"{SILVER}.api_flood_alerts",
        comment="PUB flood-alerts API records that carry an alert (most 2-minute records carry none), one row per "
        "alert: when PUB issued it, its circle (lat, lon, radius_km), and the description read by "
        "labels/alert_text.py (category flash_flood or flood_subsided, location, alert_ts from its 'Issued HHMM' "
        "stamp).",
    )
    @dp.expect("circle_in_singapore", in_singapore())
    @dp.expect("format_known", "category <> 'unmatched'")
    def api_flood_alerts():
        pages = spark.read.table("pub_flood_alerts").where(get("status", "int") == 200)
        records = pages.select(
            "fetched_at",
            F.explode(get("body.data.records", "array<variant>")).alias("rec"),
        )
        readings = records.select(
            "fetched_at",
            get("datetime", "string", "rec").alias("issued_text"),
            get_ts("updatedTimestamp", "rec").alias("updated_at"),
            F.explode(get("item.readings", "array<variant>", "rec")).alias("r"),
        )

        def circle(i: int):
            return F.expr(
                f"try_cast(try_variant_get(r, '$.area.circle[{i}]', 'string') AS DOUBLE)"
            )

        alerts = readings.select(
            "issued_text",
            F.expr("try_cast(issued_text AS TIMESTAMP)").alias("issued_at"),
            "updated_at",
            "fetched_at",
            get("area.areaDesc", "string", "r").alias("area_desc"),
            circle(0).alias("lat"),
            circle(1).alias("lon"),
            circle(2).alias("radius_km"),
            get("headline", "string", "r").alias("headline"),
            get("event", "string", "r").alias("event"),
            get("severity", "string", "r").alias("severity"),
            get("certainty", "string", "r").alias("certainty"),
            get("description", "string", "r").alias("description"),
            get("instruction", "string", "r").alias("instruction"),
        )
        # the same record comes back on every tick until it scrolls off the newest page
        once = Window.partitionBy("issued_text", "description").orderBy(
            F.desc("fetched_at")
        )
        latest = (
            alerts.withColumn("_n", F.row_number().over(once))
            .where("_n = 1")
            .drop("_n")
        )
        parsed = latest.withColumn(
            "p", parse_alert("description", "issued_text", F.lit(False))
        )
        place = F.get(
            "p.locations", 0
        )  # NULL, not an ANSI error, when nothing could be read
        return parsed.select(
            "issued_at",
            "updated_at",
            F.col("p.category").alias("category"),
            F.col("p.pattern").alias("pattern"),
            F.coalesce(
                place["location"], F.regexp_replace("area_desc", r",\s*Singapore$", "")
            ).alias("location"),
            place["stamp"].alias("stamp"),
            F.coalesce(place["stamp_ts"].cast("timestamp"), "issued_at").alias(
                "alert_ts"
            ),
            "lat",
            "lon",
            "radius_km",
            "area_desc",
            "headline",
            "event",
            "severity",
            "certainty",
            "description",
            "instruction",
            F.col("fetched_at").alias("last_fetched_at"),
        )
