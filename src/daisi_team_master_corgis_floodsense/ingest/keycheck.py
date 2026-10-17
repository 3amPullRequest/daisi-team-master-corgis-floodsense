"""Check the data.gov.sg API key. data.gov.sg answers a request whose key it doesn't recognise (mistyped, expired,
revoked) as if it had none, so the only sign is the rate limit: count the real-time calls one 10-second window allows,
and name the tier whose limit that is (client.RATE_LIMITS: 6 without a key, 12 dev, 30 prod).

    uv run python src/ingestion/check_key.py --profile flood_dev

Run it while nothing else uses the key or this connection: data.gov.sg counts every call against one budget.

How it measures: `realtime_limit` first makes calls until one is answered 429, so it knows the current window is
spent. It then polls until a call succeeds again (the window has reset, or it gives up after two windows), and counts
the calls that succeed from there until the next 429. That count is the allowance of a fresh window, so it can't
straddle two windows and read low. `tier_for` maps the count to the lowest tier whose limit is at or above it, and a
count that never hit a limit within MAX_CALLS (twice the prod limit) reads as prod. The check uses a plain requests
session rather than `client.Http`, whose throttling and retries would hide the very limit it is measuring. It exits
non-zero if there is no key, or if the limit is the no-key one, which means data.gov.sg doesn't recognise the key.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable

import requests

from . import cli
from .client import RATE_LIMITS, RATE_WINDOW_S, REALTIME_BASE, USER_AGENT

MAX_CALLS = 2 * RATE_LIMITS["realtime"]["prod"]  # past the highest limit


def realtime_limit(
    call: Callable[[], int],
    *,
    max_calls: int = MAX_CALLS,
    poll: float = 0.5,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """How many calls one fresh rate-limit window allows. `call` makes one real-time request and returns its HTTP
    status. Spends the current window first and waits for it to reset, so the count can't straddle two windows."""

    def limited() -> bool:
        status = call()
        if status not in (200, 429):
            raise RuntimeError(f"data.gov.sg answered HTTP {status}")
        return status == 429

    for _ in range(max_calls):
        if limited():
            break
    else:
        return max_calls  # never limited
    deadline = clock() + 2 * RATE_WINDOW_S
    while limited():
        if clock() > deadline:
            raise TimeoutError("the rate-limit window didn't reset")
        sleep(poll)
    allowed = 1  # the call that found the new window
    while allowed < max_calls and not limited():
        allowed += 1
    return allowed


def tier_for(limit: int) -> str:
    """The tier whose real-time limit is the lowest at or above `limit`."""
    tiers = sorted(RATE_LIMITS["realtime"].items(), key=lambda kv: kv[1])
    return next((tier for tier, cap in tiers if limit <= cap), tiers[-1][0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="~/.databrickscfg profile, to read the key from the secret scope",
    )
    parser.add_argument(
        "--secret-scope",
        default=os.environ.get("FLOODSENSE_SECRET_SCOPE", cli.SECRET_SCOPE),
        help=f"Databricks secret scope holding '{cli.API_KEY_SECRET}' (default %(default)s)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    key = cli.resolve_api_key(args.secret_scope, args.profile)
    if not key:
        print(
            f"No key found: set ${cli.API_KEY_ENV}, or store one in secret scope {args.secret_scope!r} "
            "and pass --profile."
        )
        return 1
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "x-api-key": key})

    def call() -> int:
        return session.get(f"{REALTIME_BASE}/rainfall", timeout=30).status_code

    limit = realtime_limit(call)
    tier = tier_for(limit)
    print(f"{limit} real-time calls per {RATE_WINDOW_S:g} s: ", end="")
    if tier == "none":
        print(
            "data.gov.sg doesn't recognise the key (that is the no-key limit). Check it was copied whole and "
            "hasn't expired (the data.gov.sg dashboard shows its expiry), then store it again."
        )
        return 1
    if tier == "dev":
        print("a dev key, the tier runs assume by default.")
    else:
        print(
            f"a {tier} key. Runs assume dev unless told: pass --key-tier {tier} or set "
            f"DATAGOVSG_KEY_TIER={tier} to use its full rate."
        )
    return 0
