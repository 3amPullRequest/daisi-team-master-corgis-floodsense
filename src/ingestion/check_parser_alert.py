"""Send a test parser alert: the email the developers get when PUB posts something no pattern reads, end to end.

It runs the live fallback on a made-up post in a format PUB has never used: Jev is asked once (landed under
telegram_jev_parses/test/ like every other use; with the Jev preview off, that is a free 404, which the email reports
as Jev failing), and the alert goes out by SMTP as the account in the secret scope, marked TEST:

    uv run python src/ingestion/check_parser_alert.py --landing /Volumes/flood_dev/dev_<you>_bronze/landing \
        --profile flood_dev --email-to you@example.com

This file holds no logic of its own. It puts src/ on sys.path and calls `check` of
daisi_team_master_corgis_floodsense.ingest.fallback. The exit status is non-zero when the email can't be sent, or when
Jev answers but misreads the post.
"""

import inspect
import sys
from pathlib import Path

# Put src/ on sys.path. Databricks serverless exec()s python_file tasks, so __file__ is unset there;
# the running code object still knows its own path.
sys.path.insert(
    0, str(Path(inspect.getfile(inspect.currentframe())).resolve().parents[1])
)

from daisi_team_master_corgis_floodsense.ingest.fallback import check

status = check()
if status:
    sys.exit(status)  # non-zero: no email sent, or Jev misread the post
