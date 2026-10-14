"""Static context layers (URA land use 2019/2025, PUB water-level sensors, flood-prone area, SLA benchmarks).

uv run python src/ingestion/fetch_static.py --landing /Volumes/flood_dev/dev_<you>_bronze/landing --profile flood_dev
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ingest.static import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the run (or Databricks task) failed
