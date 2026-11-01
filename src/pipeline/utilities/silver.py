"""What the silver tables share: their schema, which bronze folders have landed, VARIANT accessors, the floodsense
parsers as Spark UDFs, and the hand-kept seed files in data/static/.

- Where things are. `schema` is the silver schema, set in the pipeline's configuration, and `landed` says which of the
  bronze tables have a landing folder, so a silver table is only defined from sources that exist and an update never
  fails for a source not fetched yet.
- Reading bronze. Bronze rows hold the API's JSON as a VARIANT, and `get` and `get_ts` read a typed field from it, or
  NULL when it is missing or of another type, never an error. `in_singapore` and `distance_km` are the coordinate
  check (a box around the island and its islets) and the haversine distance (as labels/locations.haversine_km) that
  silver uses to place points; the distance is defined in the floodsense package's features.py, which flood-live shares.
- The floodsense parsers. `parse_alert`, `read_jev`, `location_key`, `name_match` and `svy21_to_wgs84` are the
  pure-Python functions of the labels package wrapped as UDFs, so Spark reads alerts exactly as the fetchers and tests do. Their
  modules are registered with cloudpickle to travel by value: the executors can't import this repository's package.
- Seeds. `seed` reads a CSV kept by hand in data/static/ (the Telegram label fixes, the gazetteer fixes), with its
  columns cast from text, or an empty table when the file is missing.
"""

import csv
import os
from pathlib import Path

from pyspark import cloudpickle
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from daisi_team_master_corgis_floodsense.features import (  # noqa: F401  (silver and gold import it from here)
    distance_km,
)
from daisi_team_master_corgis_floodsense.labels import alert_text, jev, locations

# The UDFs' code travels with them (pickled by value), so the executors don't need src/ on their path.
cloudpickle.register_pickle_by_value(alert_text)
cloudpickle.register_pickle_by_value(jev)
cloudpickle.register_pickle_by_value(locations)

SEEDS = Path(__file__).resolve().parents[3] / "data" / "static"
# A bounding box around Singapore, its islands included, for coordinate checks.
SG_BOUNDS = "BETWEEN 1.15 AND 1.48", "BETWEEN 103.59 AND 104.10"
TELEGRAM_LINK = "https://t.me/pubfloodalerts/"


def schema(spark: SparkSession) -> str:
    return spark.conf.get("silver_schema")


def landed(spark: SparkSession, *names: str) -> list[str]:
    """The bronze tables, among `names`, whose landing folders exist; bronze defines only those."""
    root = spark.conf.get("landing_path")
    return [name for name in names if os.path.isdir(f"{root}/{name}")]


def get(path: str, type_: str, column: str = "payload") -> Column:
    """A typed field of a VARIANT column, or NULL when it's missing or of another type."""
    return F.expr(f"try_variant_get({column}, '$.{path}', '{type_}')")


def get_ts(path: str, column: str = "payload") -> Column:
    """An ISO timestamp string (with its offset) of a VARIANT column, as a TIMESTAMP."""
    return F.expr(
        f"try_cast(try_variant_get({column}, '$.{path}', 'string') AS TIMESTAMP)"
    )


def in_singapore(lat: str = "lat", lon: str = "lon") -> str:
    return f"{lat} {SG_BOUNDS[0]} AND {lon} {SG_BOUNDS[1]}"


# --- the floodsense parsers, as UDFs ----------------------------------------------------------------------------------
# Only functions of the modules registered above: a function defined here would be pickled by reference, and the
# executors can't import this package. Those modules import each other's names, never each other: a module object is
# pickled whole, __loader__ included, which on Databricks belongs to dbruntime and won't unpickle on the executors
# (tests/test_silver_udfs.py repeats the list above). Filter out NULL inputs first where a function doesn't take them.

parse_alert = F.udf(alert_text.parse_row, alert_text.ROW_SCHEMA)
read_jev = F.udf(
    jev.read_row, alert_text.ROW_SCHEMA
)  # a landed Jev call's answers, read as parse_alert reads a post
location_key = F.udf(locations.location_key, "string")
name_match = F.udf(locations.name_match, "double")
svy21_to_wgs84 = F.udf(locations.svy21_to_wgs84, "struct<lat: double, lon: double>")


# --- seeds -----------------------------------------------------------------------------------------------------------


def seed(spark: SparkSession, name: str, columns: dict[str, str]) -> DataFrame:
    """A CSV kept by hand in data/static/<name>.csv (a header row; lines starting with # are notes), with `columns`
    ({name: Spark type}) cast from its text; empty when the file is missing."""
    path = SEEDS / f"{name}.csv"
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            rows = [
                tuple((row.get(c) or "").strip() or None for c in columns)
                for row in reader
            ]
    text = spark.createDataFrame(rows, ", ".join(f"{c} STRING" for c in columns))
    return text.select(
        *(F.expr(f"try_cast({c} AS {t})").alias(c) for c, t in columns.items())
    )
