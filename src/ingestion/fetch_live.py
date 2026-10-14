"""Live ingestion tick: the newest NEA/PUB API pages, new Telegram posts, radar frames and tides into the landing volume.

On Databricks this is the `fetch` task of the flood-live job. Outside Databricks (Free Edition can't reach the APIs),
run it in a loop and upload with the Files API:

    uv run python src/ingestion/fetch_live.py --landing /Volumes/flood_dev/dev_<you>_bronze/landing \
        --profile flood_dev --every 5m --daily-repair --trigger-job <flood-live job id>
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ingest.live import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the run (or Databricks task) failed
