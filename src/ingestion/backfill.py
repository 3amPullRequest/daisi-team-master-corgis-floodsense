"""Historical backfill from 2021 up to yesterday; every subcommand resumes where it stopped.

csv               data.gov.sg yearly CSVs for published years (2021-2024 today)
api-days          v2 API day pages for every later day, through yesterday
telegram          the whole PUB Flood Alerts Telegram channel
telegram-wayback  posts PUB deleted from the channel, from web archive captures (run it on a laptop)
uhslc             UHSLC hourly sea level at Tanjong Pagar

uv run python src/ingestion/backfill.py api-days --landing /Volumes/flood_dev/dev_<you>_bronze/landing --profile flood_dev
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ingest.history import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the run (or Databricks task) failed
