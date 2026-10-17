"""Flags and setup shared by the src/ingestion/*.py entry points.

Each entry point is a thin script over a `main` in this package, and every `main` starts the same way:
`add_common_args` gives it the shared flags, and `setup` turns them into the two things a fetcher needs, an `Http`
client and a `Sink` to land files in.

- Where files go. --landing (or $FLOODSENSE_LANDING) is the landing root, and --sink picks between writing to it as a
  folder and uploading with the Files API (sinks.make_sink; "auto" decides by whether the path exists here).
- Which key. `resolve_api_key` takes the data.gov.sg key from $DATAGOVSG_API_KEY, else from the `floodsense`
  Databricks secret scope (read through dbutils inside a Databricks job, through the workspace API as --profile
  elsewhere, so jobs and laptop runs share one key), and otherwise carries on without one at the lowest rate limit.
  It logs where the key came from, never the key. --key-tier says which rate limit the key has, since nothing in a key
  reveals that (keycheck.py measures it).
- Who ran it. --runner is written into every landed line (`databricks-job`, or `local:<host>`), so any file can be
  traced to the process that fetched it.

`parse_duration` reads the '90s', '5m', '5h30m' form of --every, --offset and --max-runtime.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import socket

from .client import Http
from .sinks import Sink, make_sink

log = logging.getLogger(__name__)

API_KEY_ENV = "DATAGOVSG_API_KEY"
API_KEY_SECRET = "datagovsg-api-key"
SECRET_SCOPE = "floodsense"  # databricks.yml's `secret_scope`


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--landing",
        default=os.environ.get("FLOODSENSE_LANDING"),
        required="FLOODSENSE_LANDING" not in os.environ,
        help="landing root, e.g. /Volumes/flood_dev/dev_<you>_bronze/landing (env FLOODSENSE_LANDING)",
    )
    parser.add_argument(
        "--sink",
        choices=("auto", "local", "files-api"),
        default="auto",
        help="auto: write to the folder if it exists here, else upload with the Files API",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="~/.databrickscfg profile",
    )
    parser.add_argument(
        "--secret-scope",
        default=os.environ.get("FLOODSENSE_SECRET_SCOPE", SECRET_SCOPE),
        help=f"Databricks secret scope holding '{API_KEY_SECRET}' (default %(default)s; '' for none)",
    )
    parser.add_argument(
        "--key-tier",
        choices=("dev", "prod"),
        default=os.environ.get("DATAGOVSG_KEY_TIER", "dev"),
        help="rate-limit tier of the data.gov.sg API key",
    )
    parser.add_argument(
        "--runner", default=default_runner(), help="recorded in every landed line"
    )
    parser.add_argument("--log-level", default="INFO")


def default_runner() -> str:
    return (
        "databricks-job"
        if os.environ.get("DATABRICKS_RUNTIME_VERSION")
        else f"local:{socket.gethostname()}"
    )


def resolve_api_key(secret_scope: str | None, profile: str | None = None) -> str | None:
    """The data.gov.sg API key: $DATAGOVSG_API_KEY, else `datagovsg-api-key` in the Databricks secret scope, else None
    (unauthenticated: the lowest rate limits). On Databricks the job's dbutils reads the scope; elsewhere the
    workspace API does, as `profile`, so jobs and laptop runs share one key. Logs where the key came from, never
    the key."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        log.info("data.gov.sg API key from $%s", API_KEY_ENV)
        return key
    if not secret_scope:
        return None
    on_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    if not on_databricks and not profile:
        log.warning(
            "no data.gov.sg key: pass --profile to read secret scope %r, or set $%s",
            secret_scope,
            API_KEY_ENV,
        )
        return None
    try:
        if on_databricks:
            from databricks.sdk.runtime import dbutils
        else:
            from databricks.sdk import WorkspaceClient

            dbutils = WorkspaceClient(profile=profile).dbutils
        # a value piped into `databricks secrets put-secret` can end in a newline, which no HTTP header takes
        key = dbutils.secrets.get(secret_scope, API_KEY_SECRET).strip()
    except Exception as exc:  # noqa: BLE001 - no scope, no key, no permission: go unauthenticated
        log.warning(
            "no data.gov.sg key from secret scope %r (%s); running at the lowest rate limit",
            secret_scope,
            type(exc).__name__,
        )
        return None
    if not key:
        log.warning("secret %r in scope %r is empty", API_KEY_SECRET, secret_scope)
        return None
    log.info("data.gov.sg API key from secret scope %r", secret_scope)
    return key


def setup(args: argparse.Namespace, *, share: float) -> tuple[Http, Sink]:
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    http = Http(
        resolve_api_key(args.secret_scope, args.profile),
        key_tier=args.key_tier,
        share=share,
    )
    sink = make_sink(args.landing, args.sink, args.profile)
    log.info(
        "landing: %s | data.gov.sg rate tier: %s at %.0f%%",
        sink,
        http.tier,
        share * 100,
    )
    return http, sink


def parse_duration(text: str) -> float:
    """'90s', '5m', '1h', '5h30m' -> seconds."""
    total, pos = 0.0, 0
    for match in re.finditer(r"(\d+(?:\.\d+)?)([smh])", text.strip()):
        if match.start() != pos:
            break
        total += float(match.group(1)) * {"s": 1, "m": 60, "h": 3600}[match.group(2)]
        pos = match.end()
    if pos != len(text.strip()) or total <= 0:
        raise argparse.ArgumentTypeError(
            f"bad duration {text!r}; use e.g. 90s, 5m, 5h30m"
        )
    return total
