"""Silver lightning, from the API day backfill and the live ticks. NEA's lightning data starts on 1 Feb 2025; there are
no yearly CSVs.

NEA publishes a record every 2 minutes, about 2 minutes after it closes, listing the strikes of the 2 minutes before
it: each with its time (to the millisecond), place and type. Its reach is about 0.95-1.65 N, 103.3-104.4 E, so storms
over Johor and the Riau islands show before they reach Singapore.

- silver.lightning_strikes: every strike, once. A record lists some strikes twice (232 of 12,225 in the 50 minutes
  before the 13 Apr 2025 floods), and every live tick re-fetches the newest records, so Auto CDC keeps one per strike.
- silver.lightning_records: every record, strikes or none. Most list none, so only the records tell a quiet sky from a
  feed that was down.

Both tables are built from the same API pages (the ones that answered 200) by two views and two Auto CDC flows. The
strikes view explodes each record into its strikes and drops any that lack a time or place or whose type isn't cloud
to cloud or cloud to ground. The strikes keep `published_at`, the record's publication time, because gold counts a
strike at a tick only once NEA had published it. The records view keeps one row per record with the number of strikes
it lists. Gold counts records per window to learn whether the feed was up for the whole of it, and treats lightning as
unknown, not zero, where it wasn't (gold/weather.py).
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.silver import get, get_ts, landed, schema

SILVER = schema(spark)
TARGET = f"{SILVER}.lightning_strikes"

STRIKE_CHECKS = {
    "strike_parsed": "ts IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL",
    "type_known": "type IN ('C', 'G')",
}


def _coordinate(name: str):
    """NEA sends the strike's latitude and longitude as text."""
    return F.expr(
        f"try_cast(try_variant_get(r, '$.location.{name}', 'string') AS DOUBLE)"
    )


def _records():
    """Every 2-minute record on every API page that answered 200."""
    pages = spark.readStream.table("nea_lightning").where(get("status", "int") == 200)
    return pages.select(
        F.coalesce("fetched_at", "file_modified_at").alias("fetched_at"),
        "source_file",
        F.explode(get("body.data.records", "array<variant>")).alias("rec"),
    )


if landed(spark, "nea_lightning"):

    @dp.temporary_view(name="lightning_api_strikes")
    @dp.expect_all_or_drop(STRIKE_CHECKS)
    def lightning_api_strikes():
        """Every strike on every record; most records list none."""
        records = _records()
        strikes = records.select(
            "fetched_at",
            "source_file",
            get_ts("updatedTimestamp", "rec").alias("published_at"),
            F.explode(get("item.readings", "array<variant>", "rec")).alias("r"),
        )
        return strikes.select(
            get_ts("datetime", "r").alias("ts"),
            _coordinate("latitude").alias("lat"),
            _coordinate("longitude").alias("lon"),
            get("type", "string", "r").alias("type"),
            "published_at",
            "fetched_at",
            "source_file",
        )

    dp.create_streaming_table(
        name=TARGET,
        comment="NEA lightning observations around Singapore (about 0.95-1.65 N, 103.3-104.4 E), one row per strike, "
        "from 1 Feb 2025: its time (ts, to the millisecond), place (WGS84) and type, 'C' cloud to cloud or 'G' cloud "
        "to ground. published_at is when NEA published the 2-minute record listing it, about 2 to 4 minutes after "
        "the strike.",
        cluster_by=["ts"],
    )
    dp.create_auto_cdc_flow(
        target=TARGET,
        source="lightning_api_strikes",
        keys=["ts", "lat", "lon", "type"],
        sequence_by=F.struct("fetched_at", "source_file"),
        stored_as_scd_type=1,
        name="lightning_api_strikes_into_lightning_strikes",
    )

    @dp.temporary_view(name="lightning_api_records")
    @dp.expect_or_drop("record_parsed", "record_ts IS NOT NULL")
    def lightning_api_records():
        records = _records()
        return records.select(
            get_ts("datetime", "rec").alias("record_ts"),
            get_ts("updatedTimestamp", "rec").alias("published_at"),
            F.size(get("item.readings", "array<variant>", "rec")).alias("listed"),
            "fetched_at",
            "source_file",
        )

    dp.create_streaming_table(
        name=f"{SILVER}.lightning_records",
        comment="NEA's 2-minute lightning records, one row per record whether it lists strikes or not, from 1 Feb "
        "2025: record_ts closes the 2 minutes it covers, published_at is when NEA published it, and listed counts "
        "the strikes it lists (some twice). A record that is missing means NEA's feed was down, not a quiet sky.",
        cluster_by=["record_ts"],
    )
    dp.create_auto_cdc_flow(
        target=f"{SILVER}.lightning_records",
        source="lightning_api_records",
        keys=["record_ts"],
        sequence_by=F.struct("fetched_at", "source_file"),
        stored_as_scd_type=1,
        name="lightning_api_records_into_lightning_records",
    )
