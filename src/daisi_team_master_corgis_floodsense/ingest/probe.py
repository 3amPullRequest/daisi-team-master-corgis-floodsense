"""Which ingestion hosts this machine can reach. Databricks Free Edition limits outbound internet to a set of
trusted domains, so run this on serverless first: it decides whether the fetchers can run as Databricks jobs
("native") or must run outside and upload with the Files API ("local").

`probe` makes one streamed GET to a representative URL of each host in TARGETS and reads only the status line, since
the question is whether the network path is open, so any HTTP answer, even a 403 or 404, counts as reachable. A
failure is told apart as a timeout, a DNS failure or another connection error, because they point at different fixes.
TARGETS marks which hosts the fetchers need to run natively (data.gov.sg's APIs and file store, NEA's radar store, the
Telegram channel); the others (NEA tides, UHSLC, OneMap, and PyPI as a control that proves the machine has some
internet) are reported but don't decide the verdict. `main` prints one line per host and the verdict, native if every
needed host answered, else local, naming the ones that didn't.
"""

from __future__ import annotations

import json
import time

import requests

from .client import USER_AGENT
from .sources import NEA_TIDES_URL, TELEGRAM_URL, UHSLC_URL

TARGETS = (
    (
        "data.gov.sg real-time API",
        "https://api-open.data.gov.sg/v2/real-time/api/rainfall",
        True,
    ),
    (
        "data.gov.sg metadata API",
        "https://api-production.data.gov.sg/v2/public/api/collections/2279/metadata",
        True,
    ),
    (
        "data.gov.sg dataset files (S3)",
        "https://s3.ap-southeast-1.amazonaws.com/blobs.data.gov.sg/",
        True,
    ),
    (
        "NEA radar frames (S3)",
        "https://dgs-backend-app-assets-production.s3.ap-southeast-1.amazonaws.com/",
        True,
    ),
    ("PUB Telegram channel", TELEGRAM_URL, True),
    ("NEA tide timings", NEA_TIDES_URL, False),
    ("UHSLC sea level", UHSLC_URL, False),
    (
        "OneMap search (silver geocoding)",
        "https://www.onemap.gov.sg/api/common/elastic/search?searchVal=Bishan&returnGeom=Y&getAddrDetails=N",
        False,
    ),
    ("PyPI (control)", "https://pypi.org/simple/requests/", False),
)


def probe(timeout: float = 15.0) -> list[dict]:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    results = []
    for label, url, needed in TARGETS:
        t0 = time.monotonic()
        try:
            with session.get(url, timeout=timeout, stream=True) as resp:
                # any HTTP answer means the network path is open
                outcome, status = "reachable", resp.status_code
        except requests.Timeout:
            outcome, status = "timeout", None
        except requests.ConnectionError as exc:
            text = str(exc)
            dns = any(
                s in text
                for s in (
                    "NameResolution",
                    "Name or service not known",
                    "name resolution",
                    "nodename nor servname",
                )
            )
            outcome, status = ("dns_failure" if dns else "connection_error"), None
        results.append(
            {
                "target": label,
                "url": url,
                "needed_for_native": needed,
                "outcome": outcome,
                "http_status": status,
                "ms": round((time.monotonic() - t0) * 1000),
            }
        )
    return results


def main() -> int:
    results = probe()
    for r in results:
        status = f"HTTP {r['http_status']}" if r["http_status"] else ""
        print(f"{r['outcome']:<17} {status:<9} {r['ms']:>6} ms  {r['target']}")
    blocked = [
        r["target"]
        for r in results
        if r["needed_for_native"] and r["outcome"] != "reachable"
    ]
    print(
        "VERDICT:",
        "native (fetchers can run as Databricks jobs)"
        if not blocked
        else f"local (blocked: {', '.join(blocked)})",
    )
    print(json.dumps(results))
    return 0
