"""Publish the live model's scoreboard as tables for the agencies' dashboard and Genie space (ml/publish.py).

Runs as the publish task of the flood-scoreboard job (resources/scoreboard.job.yml), whenever @champion moves.
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ml.publish import main

status = main()
if status:
    sys.exit(status)
