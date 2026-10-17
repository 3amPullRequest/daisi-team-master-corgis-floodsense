"""Print which ingestion hosts are reachable from here, and whether the fetchers can run as Databricks jobs.

Run it once on Databricks serverless (Free Edition restricts outbound internet) before choosing a run mode.

This file holds no logic of its own. It puts src/ on sys.path and calls `main` of
daisi_team_master_corgis_floodsense.ingest.probe, which tries each host the fetchers need and prints a verdict: native
(every needed host is reachable, so the fetchers can run as Databricks jobs) or local (run them elsewhere and upload).
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ingest.probe import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the run (or Databricks task) failed
