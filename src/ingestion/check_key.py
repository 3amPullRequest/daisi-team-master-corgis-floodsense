"""Check the data.gov.sg API key: count the real-time calls one 10-second window allows, and name the tier.

data.gov.sg treats a key it doesn't recognise as no key, so this is the only way to tell. Run it after storing a key:

    uv run python src/ingestion/check_key.py --profile flood_dev

This file holds no logic of its own. It puts src/ on sys.path and calls `main` of
daisi_team_master_corgis_floodsense.ingest.keycheck, which counts how many calls one rate-limit window allows and names
the key's tier from that. The exit status is non-zero when there is no key or data.gov.sg doesn't recognise it.
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ingest.keycheck import main

status = main()
if status:
    sys.exit(status)  # non-zero: no key, or one data.gov.sg doesn't recognise
