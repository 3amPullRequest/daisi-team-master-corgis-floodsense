"""bronze.ingest_heartbeat: one row per ingestion tick (per-source status, record counts, latency), for freshness checks."""

import os

from utilities.bronze_sources import HEARTBEAT_COMMENT, HEARTBEAT_TABLE
from utilities.landing_readers import json_table

LANDING = spark.conf.get("landing_path")

if os.path.isdir(f"{LANDING}/{HEARTBEAT_TABLE}"):  # defined from the first tick on
    json_table(spark, LANDING, HEARTBEAT_TABLE, HEARTBEAT_COMMENT)
