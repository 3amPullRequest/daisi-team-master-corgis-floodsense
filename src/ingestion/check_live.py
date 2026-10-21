"""Is FloodSense live? Fails when the newest tick, reading or prediction is stale (freshness.py), so the flood-watchdog
job emails its owner.

uv run python src/ingestion/check_live.py --catalog flood_dev --bronze-schema dev_<you>_bronze \
    --silver-schema dev_<you>_silver --ml-schema dev_<you>_ml
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.freshness import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the run (or Databricks task) failed
