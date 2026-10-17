"""Bronze streaming tables: Auto Loader appends every file landed under landing/<table>/ as it arrives, unchanged.

A table is defined once its landing folder exists, so a source that hasn't been fetched yet (say, before the first
backfill) doesn't fail the whole update; it appears on the first update after its files land.

Bronze is the raw layer and does no interpreting. It holds every copy of everything that ever landed, repeats included
(the same reading arrives on many ticks), and leaves deduplication and meaning to silver. This file loops over the two
catalogues in utilities/bronze_sources.py. A JSON_TABLES folder holds JSON Lines, and each line becomes one row whose
whole content is the `payload` column (a VARIANT), so a change in an API's shape never breaks the load. A CSV_TABLES
folder holds CSVs read with an explicit all-text schema, so nothing is inferred, with any column the schema doesn't
know kept in `_rescued_data`. Every row also records the file it came from and when it was ingested
(utilities/landing_readers.py), which silver uses to tell a newer copy of a reading from an older one.
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
