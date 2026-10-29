"""Check a registered pair and show what it said before every real flash flood (ml/verify.py).

Runs as the verify task of the flood-verify job (resources/verify.job.yml), after flood-train registers a pair.
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ml.verify import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the Databricks task failed: a check failed
