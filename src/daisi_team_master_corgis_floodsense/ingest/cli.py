"""Flags and setup shared by the src/ingestion/*.py entry points."""

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
        "--secret-scope", help=f"Databricks secret scope holding '{API_KEY_SECRET}'"
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


def resolve_api_key(secret_scope: str | None) -> str | None:
    """$DATAGOVSG_API_KEY, else the Databricks secret, else None (unauthenticated: the lowest rate limits)."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key
    if secret_scope:
        try:
            from databricks.sdk.runtime import dbutils

            return dbutils.secrets.get(secret_scope, API_KEY_SECRET)
        except Exception as exc:  # noqa: BLE001 - no scope, no permission, or not on Databricks: go unauthenticated
            log.warning(
                "no data.gov.sg key from secret scope %r (%s)",
                secret_scope,
                type(exc).__name__,
            )
    return None


def setup(args: argparse.Namespace, *, share: float) -> tuple[Http, Sink]:
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    http = Http(resolve_api_key(args.secret_scope), key_tier=args.key_tier, share=share)
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
