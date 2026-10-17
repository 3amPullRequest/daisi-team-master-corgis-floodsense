"""HTTP access for every ingestion source: retries, User-Agent, and data.gov.sg rate limits.

Every fetcher goes through one `Http` object, which does three jobs so the fetchers don't have to:

- Rate limits. A call made with `bucket=` ("realtime" for the v2 real-time API, "download" for dataset files) is a
  data.gov.sg call: it is throttled by a sliding-window RateLimiter and carries the API key. Calls to any other host
  (S3, t.me, NEA, UHSLC, OneMap, the web archive) are neither throttled here nor shown the key. RATE_LIMITS holds the
  calls allowed per 10 seconds by key tier (none, dev, prod), and each limiter is sized to a `share` of that allowance
  (a live tick takes 80%, a backfill 60%), which leaves room for another client on the same key. The budget is one
  per client across all data.gov.sg hosts, so metadata calls alone can make the real-time API answer 429.
- Retries. A dropped connection, a timeout, a 5xx or a 429 is retried after a jittered wait: the server's own
  Retry-After where it sends one, 10 seconds for a 429 (what data.gov.sg asks for), else capped exponential backoff.
  Any 2xx is success, and a caller can name further statuses to be handed back (`accept`). After `max_attempts` the
  failure is raised, as HttpError for an HTTP status. A 429 under a key is logged once, since it usually means
  data.gov.sg didn't recognise the key and treated the client as having none.
- Identity. Every request names itself in the User-Agent, because data.gov.sg answers the default Python agent with 403.

The data.gov.sg endpoints sit on top: `realtime` (one page), `iter_day_pages` (every page of one SGT day, newest
first, following the paginationToken until a page that isn't 200), `collection_years` and `dataset_metadata` (which
dataset holds each year of a "Historical ..." collection), and `dataset_file_url` (the download handshake: initiate it
for a CSV, then poll until a presigned S3 URL is ready). A date with no data comes back as a 404 carrying body code
17; `get_api_json` returns that as an ordinary response (`ApiResponse.no_data`) so a backfill can land it as the
day's "nothing here" marker instead of failing.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

log = logging.getLogger(__name__)

# data.gov.sg answers the default `Python-urllib/*` agent with HTTP 403, so every request names itself.
USER_AGENT = "floodsense-ingest/0.1 (+https://github.com/3ampullrequest/daisi-team-master-corgis-floodsense)"

REALTIME_BASE = "https://api-open.data.gov.sg/v2/real-time/api"
DATASETS_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
METADATA_BASE = "https://api-production.data.gov.sg/v2/public/api"

# Requests allowed per 10 s window by key tier (guide.data.gov.sg "API Rate Limits"). The budget is shared across
# data.gov.sg hosts for one client: metadata calls alone are enough to make the real-time API return 429.
RATE_LIMITS = {
    "realtime": {"none": 6, "dev": 12, "prod": 30},
    "download": {"none": 2, "dev": 4, "prod": 10},
}
RATE_WINDOW_S = 10.0

# Body code data.gov.sg sends (with HTTP 404) when a date has no data.
NO_DATA_CODE = 17


class RateLimiter:
    """Allows at most `max_calls` acquisitions in any sliding `window` of seconds."""

    def __init__(
        self,
        max_calls: int,
        window: float = RATE_WINDOW_S,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.max_calls = max(1, max_calls)
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.window:
                self._calls.popleft()
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return
            self._sleep(self.window - (now - self._calls[0]))


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:300]!r}")
        self.url = url
        self.status = status


@dataclass
class ApiResponse:
    """A data.gov.sg JSON response. `url` excludes the query string; `params` holds it (never the API key)."""

    url: str
    params: dict[str, str]
    status: int
    body: Any

    @property
    def no_data(self) -> bool:
        return (
            self.status == 404
            and isinstance(self.body, dict)
            and self.body.get("code") == NO_DATA_CODE
        )

    @property
    def data(self) -> dict:
        data = self.body.get("data") if isinstance(self.body, dict) else None
        return data if isinstance(data, dict) else {}


class Http:
    """A requests session with retries. Passing `bucket=` marks a data.gov.sg call: it is throttled against that
    rate-limit bucket and carries the API key. Other hosts (S3, t.me, NEA, UHSLC) never see the key."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        key_tier: str = "dev",
        share: float = 0.8,
        session: requests.Session | None = None,
        timeout: float = 60.0,
        max_attempts: int = 8,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.tier = key_tier if api_key else "none"
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.limiters = {
            bucket: RateLimiter(
                int(limits[self.tier] * share), clock=clock, sleep=sleep
            )
            for bucket, limits in RATE_LIMITS.items()
        }
        self._warned_limited = False

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        *,
        bucket: str | None = None,
        stream: bool = False,
        accept: tuple[int, ...] = (),
    ) -> requests.Response:
        headers = {"x-api-key": self.api_key} if bucket and self.api_key else {}
        for attempt in range(1, self.max_attempts + 1):
            if bucket:
                self.limiters[bucket].acquire()
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=stream,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_attempts:
                    raise
                delay = _backoff(attempt)
                log.warning(
                    "%s on %s; retry %d in %.0fs",
                    type(exc).__name__,
                    url,
                    attempt,
                    delay,
                )
                self.sleep(delay)
                continue
            # any 2xx is success: poll-download answers 201
            if 200 <= resp.status_code < 300 or resp.status_code in accept:
                return resp
            if resp.status_code == 429 and bucket and self.api_key:
                self._warn_limited()
            if (
                resp.status_code == 429 or resp.status_code >= 500
            ) and attempt < self.max_attempts:
                # data.gov.sg's 429 says "try again in 10 seconds"
                delay = _retry_after(resp) or (
                    10.0 if resp.status_code == 429 else _backoff(attempt)
                )
                delay += random.uniform(0, 2)
                log.warning(
                    "HTTP %d on %s; retry %d in %.0fs",
                    resp.status_code,
                    url,
                    attempt,
                    delay,
                )
                resp.close()
                self.sleep(delay)
                continue
            raise HttpError(url, resp.status_code, resp.text)
        raise AssertionError("unreachable")

    def _warn_limited(self) -> None:
        """Once per client: a 429 below the key's limit. data.gov.sg treats a key it doesn't recognise as no key."""
        if not self._warned_limited:
            self._warned_limited = True
            log.warning(
                "data.gov.sg rate-limited this client below the %s-key limit: either the key isn't recognised "
                "(mistyped or expired; check it with src/ingestion/check_key.py), or another client shares it",
                self.tier,
            )

    def get_api_json(
        self,
        url: str,
        params: dict[str, str] | None = None,
        *,
        bucket: str = "realtime",
    ) -> ApiResponse:
        """GET a data.gov.sg JSON endpoint. A 'no data for this date' 404 comes back as a response, not an error."""
        resp = self.get(url, params, bucket=bucket, accept=(404,))
        body = resp.json()
        result = ApiResponse(
            url=url, params=dict(params or {}), status=resp.status_code, body=body
        )
        if resp.status_code == 404 and not result.no_data:
            raise HttpError(url, 404, resp.text)
        return result


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0**attempt) + random.uniform(0, 1)


def _retry_after(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    try:
        return float(value) if value else None
    except ValueError:
        return None


# --- data.gov.sg endpoints ---------------------------------------------------------------------------------------


def realtime(
    http: Http, endpoint: str, params: dict[str, str] | None = None
) -> ApiResponse:
    return http.get_api_json(f"{REALTIME_BASE}/{endpoint}", params)


def iter_day_pages(
    http: Http,
    endpoint: str,
    day: date,
    params: dict[str, str] | None = None,
    *,
    max_pages: int = 200,
) -> Iterator[ApiResponse]:
    """Every page of one SGT day, newest first (25 records a page). Ends after a non-200 page (404 = no data)."""
    query = {**(params or {}), "date": day.isoformat()}
    for _ in range(max_pages):
        resp = realtime(http, endpoint, query)
        yield resp
        token = resp.data.get("paginationToken") if resp.status == 200 else None
        if not token:
            return
        query = {**query, "paginationToken": token}
    raise RuntimeError(f"{endpoint} {day}: more than {max_pages} pages")


def dataset_metadata(http: Http, dataset_id: str) -> dict:
    return http.get_api_json(f"{METADATA_BASE}/datasets/{dataset_id}/metadata").data


def collection_years(http: Http, collection_id: int) -> dict[int, str]:
    """Year -> dataset id for a yearly collection such as "Historical Rainfall across Singapore (2024)"."""
    meta = http.get_api_json(
        f"{METADATA_BASE}/collections/{collection_id}/metadata"
    ).data
    years: dict[int, str] = {}
    for dataset_id in (meta.get("collectionMetadata") or {}).get("childDatasets") or []:
        match = re.search(
            r"\((\d{4})\)\s*$", dataset_metadata(http, dataset_id).get("name") or ""
        )
        if match:
            years[int(match.group(1))] = dataset_id
    return years


def dataset_file_url(
    http: Http,
    dataset_id: str,
    *,
    initiate: bool,
    poll_every: float = 3.0,
    max_polls: int = 100,
) -> str:
    """Presigned S3 URL of a dataset file. CSV datasets must be initiated first; GeoJSON/XLSX/PDF can be polled directly."""
    if initiate:
        http.get_api_json(
            f"{DATASETS_BASE}/{dataset_id}/initiate-download", bucket="download"
        )
    for _ in range(max_polls):
        data = http.get_api_json(
            f"{DATASETS_BASE}/{dataset_id}/poll-download", bucket="download"
        ).data
        status = data.get("status")
        if data.get("url") and status in (None, "DOWNLOAD_SUCCESS"):
            return data["url"]
        if status and "FAIL" in status:
            raise RuntimeError(f"{dataset_id}: download {status}")
        http.sleep(poll_every)
    raise TimeoutError(f"{dataset_id}: download not ready after {max_polls} polls")
