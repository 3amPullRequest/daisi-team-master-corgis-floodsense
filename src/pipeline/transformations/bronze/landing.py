"""Bronze streaming tables: Auto Loader appends every file landed under landing/<table>/ as it arrives, unchanged.

A table is defined once its landing folder exists, so a source that hasn't been fetched yet (say, before the first
backfill) doesn't fail the whole update; it appears on the first update after its files land.
"""

import os

from utilities.bronze_sources import CSV_TABLES, JSON_TABLES
from utilities.landing_readers import csv_table, json_table

LANDING = spark.conf.get("landing_path")

for name, comment in JSON_TABLES.items():
    if os.path.isdir(f"{LANDING}/{name}"):
        json_table(spark, LANDING, name, comment)

for name, options in CSV_TABLES.items():
    if os.path.isdir(f"{LANDING}/{name}"):
        csv_table(spark, LANDING, name, **options)
