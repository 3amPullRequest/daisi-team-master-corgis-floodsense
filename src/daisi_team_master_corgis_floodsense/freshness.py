"""Is FloodSense live? The flood-watchdog job's check: how old the newest tick, reading and prediction are, failing
(so the job emails its owner) when any is older than its limit.

flood-live can stop without failing: its schedule paused by a dev redeploy, the Free Edition quota spent for the day, a
run hanging until its timeout, or NEA's feed stalling while every request still answers 200. Each shows here as
something going stale:

- `tick`: the newest bronze.ingest_heartbeat line (the fetch ran);
- `core sources`: the newest tick in which every core source (sources.CORE_SOURCES) answered;
- `rain reading`: the newest silver.rainfall_5min timestamp (NEA's feed moved, and flood-etl streamed it);
- `prediction`: the newest ml.predictions tick (flood-live's score task wrote it).

    uv run python src/ingestion/check_live.py --catalog flood_dev --bronze-schema dev_<you>_bronze \\
        --silver-schema dev_<you>_silver --ml-schema dev_<you>_ml   # on Databricks: the flood-watchdog job
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from .ingest.sources import CORE_SOURCES

log = logging.getLogger(__name__)

# A 5-minute cadence whose runs can skip one another (a refresh takes about 5 minutes), plus NEA's own reading lag:
# twenty minutes without a tick, or half an hour without a reading or a prediction, is an outage, not a slow run.
LIMITS_MIN = {"tick": 20, "core sources": 20, "rain reading": 30, "prediction": 30}


def problems(
    now: pd.Timestamp, newest: dict[str, pd.Timestamp | None], limits=LIMITS_MIN
) -> list[str]:
    """What is stale, in words; empty when FloodSense is live. Times are naive SGT."""
    out = []
    for name, limit in limits.items():
        ts = newest.get(name)
        if ts is None or pd.isna(ts):
            out.append(f"no {name} at all")
            continue
        age = (now - ts) / pd.Timedelta(minutes=1)
        if age > limit:
            out.append(f"newest {name} is {age:.0f} min old (limit {limit}): {ts}")
    return out


def newest_times(
    spark, bronze: str, silver: str, ml: str
) -> dict[str, pd.Timestamp | None]:
    core = ", ".join(f"'{s}'" for s in CORE_SOURCES)
    queries = {
        "tick": f"SELECT max(fetched_at) FROM {bronze}.ingest_heartbeat",
        # a tick counts when every core source in it answered
        "core sources": f"""
            SELECT max(fetched_at) FROM (
              SELECT h.fetched_at, count_if(r.value:ok::boolean) AS ok, count(*) AS n
              FROM {bronze}.ingest_heartbeat AS h, LATERAL variant_explode(h.payload:results) AS r
              WHERE r.value:source::string IN ({core}) GROUP BY h.fetched_at
            ) WHERE ok = n AND n = {len(CORE_SOURCES)}""",
        "rain reading": f"SELECT max(ts) FROM {silver}.rainfall_5min",
        "prediction": f"SELECT max(ts) FROM {ml}.predictions",
    }
    out = {}
    for name, sql in queries.items():
        try:
            value = spark.sql(sql).collect()[0][0]
        except Exception as exc:  # noqa: BLE001 - a missing table is an answer: nothing landed there yet
            log.warning("%s: %s", name, exc)
            value = None
        out[name] = None if value is None else pd.Timestamp(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ml-schema", default="ml")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set(
        "spark.sql.session.timeZone", "Asia/Singapore"
    )  # naive times are SGT
    newest = newest_times(
        spark,
        *(
            f"{args.catalog}.{s}"
            for s in (args.bronze_schema, args.silver_schema, args.ml_schema)
        ),
    )
    now = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None)
    found = problems(now, newest)
    log.info("newest: %s", json.dumps({k: str(v) for k, v in newest.items()}))
    if found:
        # a failed run is what emails the job's owner
        raise SystemExit("FloodSense is not live: " + "; ".join(found))
    log.info("FloodSense is live")
    return 0
