"""Helpers for the flood-etl pipeline.

Databricks puts only the pipeline root (src/pipeline) on sys.path. Importing this package also puts src/ there, so
pipeline code reads alerts with the same floodsense package (labels, ml.contract) that the jobs and tests use.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.append(_SRC)
