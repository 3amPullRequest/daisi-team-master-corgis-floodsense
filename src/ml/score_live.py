"""Score the newest ticks for all 47 areas and append them to ml.predictions (ml/live.py).

Runs as the score task of the flood-live job (resources/live.job.yml), after each refresh of silver.
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ml.live import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the Databricks task failed
