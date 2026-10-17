"""bronze.ingest_heartbeat: one row per ingestion tick (per-source status, record counts, latency), for freshness checks.

The live fetcher (ingest/live.py) ends every tick by landing one heartbeat line: whether each source answered, how many
records it returned and how long it took. This table is that line streamed in like any other JSON Lines source, kept
apart from the data sources in bronze_sources.py because it describes the fetcher and not a dataset. Nothing here
interprets it. It is how the pipeline knows a source was up at a given time: gold.label_coverage reads it to learn how
far the Telegram channel was being watched, which separates a quiet channel from one nobody was reading. Like the other
bronze tables, it is defined only once its folder exists, from the first tick on.
"""

import os

from utilities.bronze_sources import HEARTBEAT_COMMENT, HEARTBEAT_TABLE
from utilities.landing_readers import json_table

LANDING = spark.conf.get("landing_path")

if os.path.isdir(f"{LANDING}/{HEARTBEAT_TABLE}"):  # defined from the first tick on
    json_table(spark, LANDING, HEARTBEAT_TABLE, HEARTBEAT_COMMENT)
