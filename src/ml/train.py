"""Train the gated LightGBM pair and the rainfall rule on gold, log them to MLflow and register the pair in Unity Catalog.

Runs as the flood-train job (resources/train.job.yml): databricks bundle run train -t dev -p flood_dev

This file holds no logic of its own. It puts src/ on sys.path and calls `main` of
daisi_team_master_corgis_floodsense.ml.train, which is where the order of work is explained: load gold, split, fit the
rule, fit the pair out of fold, judge it against the rule, log and register it, and (only on request) score the test.
The exit status marks the Databricks task failed or not.
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ml.train import main

status = main()
if status:
    sys.exit(status)  # non-zero marks the Databricks task failed
