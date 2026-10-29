"""silver.telegram_posts and silver.telegram_review: PUB's Telegram posts read by the floodsense patterns
(labels/alert_text.py), and the posts that need a person to look at them.

A post lands many times: in each channel scrape, in each web archive capture that shows it, and live. Its id is the
same under both channel usernames, so it is keyed on message_id alone, and the latest version seen wins (a capture
was seen when the archive took it, not when it was fetched).

A post no pattern matches is `unmatched`, and it waits in telegram_review. Once a person has looked, either a new
pattern reads it (a new format), or data/static/telegram_manual_labels.csv labels it, e.g. `not_relevant` for an
announcement. A manual label wins over the patterns. Until then, a live post the patterns can't read (unmatched, or
matched with no location) is read from Jev's answers (labels/jev.py), which flood-live's fetch landed in
bronze.telegram_jev_parses: when Jev was sure enough, its reading stands in, with method 'jev', so the alert reaches
silver.alerts and the app minutes after PUB posts it. The pattern takes over as soon as one reads the post.

How telegram_posts is built. Bronze holds every copy of every post, from each channel scrape, each web-archive capture
and each live tick. The table first reduces them to one row per message_id, the copy seen latest (an archive capture
counts from `captured_at`), and counts the copies and the first sighting. It then runs the post through the same parser
the fetchers use, which gives the category, the matching pattern, any problem, and each location with the moment PUB
stamped against it (and, for NEA's heavy-rain posts, the regions and the valid window). Where the parser can't read a
post, Jev's newest live answers for that id and text are read in the same shape and used instead, if Jev was sure.
Last, it joins the hand labels: a manual category overrides both and clears the problem. `origin` (archive, live or wayback), read from the
landed file's folder, is kept because gold.label_coverage needs to know which posts show when the channel's newest
page was read.

telegram_review lists the posts that need a person, with the action that resolves each: an unmatched post needs a new
format in FORMATS or a manual label, a post Jev read stands in for one until then, a post with no stamp is usable but
worth a look, and a post whose format matched but whose location could not be read needs its reader fixed. The expectations on telegram_posts (`format_known`,
`read_fully`) count those posts without failing the update.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.silver import (
    TELEGRAM_LINK,
    get,
    get_ts,
    landed,
    parse_alert,
    read_jev,
    schema,
    seed,
)

SILVER = schema(spark)
JEV = bool(landed(spark, "telegram_jev_parses"))


def jev_answers():
    """Per post id and text, Jev's newest successful answers from a live fetch (bronze.telegram_jev_parses)."""
    calls = (
        spark.read.table("telegram_jev_parses")
        .select(
            get("message_id", "bigint").alias("message_id"),
            get("text_sha256", "string").alias("text_sha256"),
            F.expr(
                "to_json(try_variant_get(payload, '$.candidates', 'variant'))"
            ).alias("jev_candidates"),
            F.expr("to_json(try_variant_get(payload, '$.answers', 'variant'))").alias(
                "jev_answers"
            ),
            get_ts("fetched_at").alias("jev_asked_at"),
            get("status", "string").alias("status"),
            get("purpose", "string").alias("purpose"),
        )
        .where("status = 'ok' AND purpose = 'live' AND jev_answers IS NOT NULL")
    )
    newest = Window.partitionBy("message_id", "text_sha256").orderBy(
        F.desc("jev_asked_at")
    )
    return (
        calls.withColumn("_n", F.row_number().over(newest))
        .where("_n = 1")
        .drop("_n", "status", "purpose")
    )


if landed(spark, "pub_telegram_posts"):

    @dp.materialized_view(
        name=f"{SILVER}.telegram_posts",
        comment="PUB Flood Alerts Telegram posts, one row per message_id (the latest version seen), classified by "
        "labels/alert_text.py: category, the pattern that matched, and each location with the time PUB stamped "
        "against it. method is 'pattern'; 'jev' where no pattern reads the post and Jev's answers "
        "(bronze.telegram_jev_parses) stand in; or 'manual' for a label kept in data/static/telegram_manual_labels.csv.",
    )
    @dp.expect("format_known", "category <> 'unmatched'")
    @dp.expect("read_fully", "problem IS NULL")
    def telegram_posts():
        copies = spark.read.table("pub_telegram_posts").select(
            get("message_id", "bigint").alias("message_id"),
            get("posted_at", "string").alias("posted_text"),
            get("text", "string").alias("text"),
            F.coalesce(get("service", "boolean"), F.lit(False)).alias("service"),
            F.coalesce(get("edited", "boolean"), F.lit(False)).alias("edited"),
            get("channel", "string").alias("channel"),
            get("channel_title", "string").alias("channel_title"),
            get("capture_url", "string").alias("capture_url"),
            F.coalesce(get_ts("captured_at"), "fetched_at").alias("seen_at"),
            F.regexp_extract("source_file", r"/pub_telegram_posts/([a-z]+)/", 1).alias(
                "origin"
            ),
            "source_file",
        )
        by_post = Window.partitionBy("message_id")
        latest = (
            copies.where("message_id IS NOT NULL")
            .withColumn("copies", F.count("*").over(by_post))
            .withColumn("first_seen", F.min("seen_at").over(by_post))
            .withColumn(
                "_n",
                F.row_number().over(
                    by_post.orderBy(F.desc("seen_at"), F.desc("source_file"))
                ),
            )
            .where("_n = 1")
        )
        parsed = latest.withColumn(
            "p", parse_alert("text", "posted_text", "service")
        ).withColumn("text_sha256", F.sha2(F.coalesce("text", F.lit("")), 256))
        if JEV:
            parsed = parsed.join(jev_answers(), ["message_id", "text_sha256"], "left")
        else:
            parsed = parsed.select(
                "*",
                *(
                    F.lit(None).cast(t).alias(c)
                    for c, t in (
                        ("jev_candidates", "string"),
                        ("jev_answers", "string"),
                        ("jev_asked_at", "timestamp"),
                    )
                ),
            )
        parsed = parsed.withColumn(
            "j",
            F.when(
                F.col("jev_answers").isNotNull(),
                read_jev("text", "posted_text", "jev_candidates", "jev_answers"),
            ),
        )

        manual = seed(
            spark,
            "telegram_manual_labels",
            {"message_id": "BIGINT", "category": "STRING", "note": "STRING"},
        ).dropDuplicates(["message_id"])
        labelled = parsed.join(
            manual.withColumnRenamed("category", "manual_category"),
            "message_id",
            "left",
        )

        # Jev stands in where the patterns can't read the post (as jev.needs_fallback) and it was sure of the category
        pattern_reads = (F.col("p.category") != "unmatched") & ~F.coalesce(
            F.col("p.problem") == "no location", F.lit(False)
        )
        use_jev = (
            F.col("manual_category").isNull()
            & ~pattern_reads
            & F.coalesce(F.col("j.category") != "unmatched", F.lit(False))
        )
        labelled = labelled.withColumn(
            "r", F.when(use_jev, F.col("j")).otherwise(F.col("p"))
        ).withColumn("use_jev", use_jev)
        locations = F.transform(
            "r.locations",
            lambda loc: F.struct(
                loc["location"].alias("location"),
                loc["stamp"].alias("stamp"),
                loc["stamp_ts"].cast("timestamp").alias("stamp_ts"),
            ),
        )
        return labelled.select(
            "message_id",
            F.expr("try_cast(posted_text AS TIMESTAMP)").alias("posted_at"),
            F.concat(F.lit(TELEGRAM_LINK), F.col("message_id").cast("string")).alias(
                "link"
            ),
            F.coalesce("manual_category", "r.category").alias("category"),
            F.when(F.col("manual_category").isNotNull(), "manual")
            .when(F.col("use_jev"), "jev")
            .otherwise("pattern")
            .alias("method"),
            F.col("r.pattern").alias("pattern"),
            F.when(F.col("manual_category").isNull(), F.col("r.problem")).alias(
                "problem"
            ),
            F.col("p.hint").alias("hint"),
            locations.alias("locations"),
            F.col("r.regions").alias("regions"),
            F.col("r.issued_ts").cast("timestamp").alias("issued_at"),
            F.col("r.valid_from").cast("timestamp").alias("valid_from"),
            F.col("r.valid_to").cast("timestamp").alias("valid_to"),
            F.get_json_object("jev_answers", "$.category.choice").alias("jev_category"),
            F.get_json_object("jev_answers", "$.category.confidence")
            .cast("double")
            .alias("jev_confidence"),
            "jev_asked_at",
            "text",
            "edited",
            "service",
            "channel",
            "channel_title",
            "origin",
            "copies",
            "first_seen",
            "seen_at",
            "capture_url",
            "source_file",
            F.col("note").alias("manual_note"),
        )

    @dp.materialized_view(
        name=f"{SILVER}.telegram_review",
        comment="Telegram posts waiting for a person: no pattern matches them (a new format, or not an alert), Jev "
        "read them in place of a pattern, or a pattern matched but couldn't read a location or a stamp. `action` says "
        "what resolves each.",
    )
    def telegram_review():
        posts = spark.read.table(f"{SILVER}.telegram_posts")
        action = (
            F.when(
                F.col("method") == "jev",
                F.lit(
                    "Jev read it in place of a pattern (its reading is in use): add its format to "
                    "labels/alert_text.py FORMATS (with the post as a test fixture), or label it in "
                    "data/static/telegram_manual_labels.csv"
                ),
            )
            .when(
                F.col("category") == "unmatched",
                F.lit(
                    "If it is an alert, add its format to labels/alert_text.py FORMATS (with the post as a test "
                    "fixture); if not, label it in data/static/telegram_manual_labels.csv"
                ),
            )
            .when(
                F.col("problem") == "no stamp",
                F.lit(
                    "Usable: its alerts are timed at the posting. Check the text for a time the pattern missed"
                ),
            )
            .otherwise(
                F.lit(
                    "The format matched but its location couldn't be read: fix the format's reader"
                )
            )
        )
        return posts.where(
            "category = 'unmatched' OR problem IS NOT NULL OR method = 'jev'"
        ).select(
            "message_id",
            "posted_at",
            "link",
            "category",
            "method",
            "pattern",
            "problem",
            "hint",
            "jev_category",
            "jev_confidence",
            action.alias("action"),
            "text",
        )
