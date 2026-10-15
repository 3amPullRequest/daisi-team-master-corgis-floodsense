"""Gold labels: which ticks' labels are known, and the labels themselves, split 80/20 in time.

- gold.label_coverage: PUB's channel history as intervals of posting time between consecutive recovered posts.
  Telegram numbers posts without gaps, so neighbours with consecutive ids prove nothing was posted between them
  ('recorded'). An id gap means posts PUB deleted before the web archive captured them, and the whole span between its
  neighbours is 'missing': the lost posts could have gone out at any moment of it. After the newest post, the channel
  is recorded up to the last time its newest page was read.
- gold.label_ticks: one row per tick. Its label window is missing if any post that could carry an alert for it
  (contract.py: posted from STAMP_AHEAD before the tick to HORIZON + POST_DELAY_MAX after) falls in a missing interval,
  or if an alert for it isn't placed in an area yet; and its split.
- gold.area_labels: per area and tick, whether a PUB alert came within the hour ('alert'), none came and the window was
  recorded ('no_alert'), or none was recorded but the window wasn't ('missing'), as scoreboard.tick_labels counts them.
"""

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.contract_tables import AREA_LABELS_SCHEMA, LABEL_TICKS_SCHEMA
from utilities.gold import (
    DAY_S,
    TICK_S,
    bronze,
    seconds,
    sgt_day_floor,
    sgt_midnight_s,
    silver,
    tick_after,
    tick_at_or_after,
    tick_before,
    ticks_between,
)
from utilities.silver import get

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.split import EMBARGO_DAYS

HORIZON_S = C.HORIZON_MINUTES * 60
REACH_S = (
    HORIZON_S + C.POST_DELAY_MAX_MINUTES * 60
)  # posts this long after a tick can carry its label
AHEAD_S = C.STAMP_AHEAD_MINUTES * 60  # ... and this long before it
CONSECUTIVE = "consecutive post ids"
HEAD_SEEN = "channel's newest page read"


def _split(tick_s):
    test_s = sgt_midnight_s(C.TEST_START)
    return (
        F.when(tick_s < test_s - EMBARGO_DAYS * DAY_S, "train")
        .when(tick_s < test_s, "embargo")
        .when(tick_s < sgt_midnight_s(C.SPLIT_END), "test")
        .otherwise("after_split")
    )


@dp.materialized_view(
    name="label_coverage",
    comment="PUB's Telegram channel as intervals of posting time between consecutive recovered posts: 'recorded' "
    "where their ids are consecutive (nothing was posted between them), 'missing' where an id gap means posts were "
    "never recovered (any moment of the interval could hold them). The last interval runs from the newest post to "
    "the last time the channel's newest page was read.",
)
@dp.expect("intervals_forward", "to_ts >= from_ts")
def label_coverage():
    posts = (
        spark.read.table(silver(spark, "telegram_posts"))
        .where("posted_at IS NOT NULL")
        .select("message_id", "posted_at", "origin", "seen_at")
    )
    by_id = Window.orderBy("message_id")
    pairs = posts.select(
        F.col("message_id").alias("after_message_id"),
        F.col("posted_at").alias("from_ts"),
        F.lead("message_id").over(by_id).alias("before_message_id"),
        F.lead("posted_at").over(by_id).alias("to_ts"),
    ).where("before_message_id IS NOT NULL")
    consecutive = F.col("before_message_id") == F.col("after_message_id") + 1
    between = pairs.select(
        "from_ts",
        "to_ts",
        F.when(consecutive, C.RECORDED).otherwise(C.MISSING).alias("coverage"),
        F.when(consecutive, CONSECUTIVE)
        .otherwise(C.POSTS_NOT_RECOVERED)
        .alias("reason"),
        "after_message_id",
        "before_message_id",
        (F.col("before_message_id") - F.col("after_message_id") - 1).alias(
            "missing_posts"
        ),
    )

    # the channel's newest page was read by every archive scrape and every live tick that reached it
    scraped = posts.where(F.col("origin").isin("archive", "live")).select(
        F.col("seen_at").alias("read_at")
    )
    ticks = (
        spark.read.table(bronze(spark, "ingest_heartbeat"))
        .select("fetched_at", F.explode(get("results", "array<variant>")).alias("r"))
        .where(
            (get("source", "string", "r") == "pub_telegram_posts")
            & get("ok", "boolean", "r")
        )
        .select(F.col("fetched_at").alias("read_at"))
    )
    newest = posts.agg(
        F.max("message_id").alias("after_message_id"),
        F.max_by("posted_at", "message_id").alias("from_ts"),
    )
    tail = (
        newest.crossJoin(
            scraped.unionByName(ticks).agg(F.max("read_at").alias("to_ts"))
        )
        .where("to_ts >= from_ts")
        .select(
            "from_ts",
            "to_ts",
            F.lit(C.RECORDED).alias("coverage"),
            F.lit(HEAD_SEEN).alias("reason"),
            "after_message_id",
            F.lit(None).cast("bigint").alias("before_message_id"),
            F.lit(0).cast("bigint").alias("missing_posts"),
        )
    )
    return between.unionByName(tail).withColumn(
        "hours", F.round((seconds("to_ts") - seconds("from_ts")) / 3600, 2)
    )


@dp.materialized_view(
    name=C.LABEL_TICKS,
    comment="One row per 5-minute tick from contract.LABELS_START to the last midnight the channel was watched past: "
    "whether the tick's label window is recorded or missing, and why, and the tick's split (contract.py: 80% of the "
    "recorded time to train, the last 20% to test, a day's embargo between). The same for every area.",
    schema=LABEL_TICKS_SCHEMA,
    cluster_by=["ts"],
)
@dp.expect("known_split", f"split IN {C.SPLITS}")
def label_ticks():
    coverage = spark.read.table("label_coverage")
    # the last midnight whose ticks' windows the channel was watched to the end of
    end_s = sgt_day_floor(F.max(seconds("to_ts")) - REACH_S)
    grid = coverage.agg(end_s.alias("end_s")).select(
        ticks_between(
            F.lit(sgt_midnight_s(C.LABELS_START)), F.col("end_s") - TICK_S
        ).alias("tick_s")
    )
    # a missing interval (from, to) hides the windows of the ticks from REACH before it to AHEAD after it
    gaps = (
        coverage.where(F.col("coverage") == C.MISSING)
        .select(
            ticks_between(
                tick_after(seconds("from_ts") - REACH_S),
                tick_before(seconds("to_ts") + AHEAD_S),
            ).alias("tick_s")
        )
        .distinct()
        .withColumn("posts_missing", F.lit(True))
    )
    # an alert not placed in an area could be any area's: it hides the windows of the hour of ticks before it
    unplaced = (
        spark.read.table(silver(spark, "alerts_unplaced"))
        .where("alert_ts IS NOT NULL")
        .select(
            ticks_between(
                tick_at_or_after(seconds("alert_ts") - HORIZON_S),
                tick_before(seconds("alert_ts")),
            ).alias("tick_s")
        )
        .distinct()
        .withColumn("unplaced", F.lit(True))
    )
    rows = grid.join(gaps, "tick_s", "left").join(unplaced, "tick_s", "left")
    reason = F.when(F.col("posts_missing"), C.POSTS_NOT_RECOVERED).when(
        F.col("unplaced"), C.UNPLACED_ALERT
    )
    return rows.select(
        F.timestamp_seconds("tick_s").alias("ts"),
        F.when(reason.isNull(), C.RECORDED).otherwise(C.MISSING).alias("label_window"),
        reason.alias("missing_reason"),
        _split(F.col("tick_s")).alias("split"),
    )


@dp.materialized_view(
    name=C.AREA_LABELS,
    comment="The labels per NEA forecast area and tick (contract.AREA_LABELS_COLUMNS): 'alert' if a PUB alert came "
    "in the area within the next hour, 'no_alert' if none came and the window was recorded, 'missing' if none was "
    "recorded but the window wasn't. Train on alert_label where it isn't null, within split = 'train'.",
    schema=AREA_LABELS_SCHEMA,
    cluster_by=["ts", "area"],
)
@dp.expect(
    "labelled_where_recorded", "alert_label IS NOT NULL OR missing_reason IS NOT NULL"
)
def area_labels():
    alerts = (
        spark.read.table(silver(spark, C.AREA_ALERTS))
        .where("alert_ts IS NOT NULL AND area IS NOT NULL")
        .select("area", "kind", seconds("alert_ts").alias("alert_s"))
    )
    # an alert at a labels the ticks t with t < a <= t + HORIZON
    hits = (
        alerts.select(
            "area",
            (F.col("kind") == C.FLASH_FLOOD).cast("int").alias("is_flood"),
            ticks_between(
                tick_at_or_after(F.col("alert_s") - HORIZON_S),
                tick_before(F.col("alert_s")),
            ).alias("tick_s"),
        )
        .groupBy("area", "tick_s")
        .agg(F.max("is_flood").alias("flood_hit"))
        .withColumn("alert_hit", F.lit(1))
    )
    areas = spark.read.table(silver(spark, "forecast_areas")).select("area")
    grid = (
        spark.read.table(C.LABEL_TICKS)
        .withColumn("tick_s", seconds("ts"))
        .crossJoin(F.broadcast(areas))
    )
    rows = grid.join(hits, ["area", "tick_s"], "left")
    recorded = F.col("label_window") == C.RECORDED
    alert_label = F.when(F.col("alert_hit") == 1, 1).when(recorded, 0)
    flood_label = F.when(F.col("flood_hit") == 1, 1).when(recorded, 0)
    status = (
        F.when(alert_label == 1, C.ALERT)
        .when(alert_label == 0, C.NO_ALERT)
        .otherwise(C.MISSING)
    )
    return rows.select(
        "area",
        "ts",
        status.alias("alert_status"),
        alert_label.cast("int").alias("alert_label"),
        flood_label.cast("int").alias("flood_label"),
        "missing_reason",
        "split",
    )
