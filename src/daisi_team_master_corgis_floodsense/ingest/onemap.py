"""OneMap searches for PUB alert locations, landed raw for silver's gazetteer (silver.location_gazetteer).

Silver places most alerts by PUB's own coordinates and its sensor names; OneMap covers the rest, chiefly the roads that
flash floods name. Each alert location in the landed Telegram posts that hasn't been searched is looked up by
`locations.search_queries`, best guess first, until one finds a place in Singapore. Every answer lands, found or not,
so a re-run asks only about new locations.

OneMap's search answers without a token, but (Oct 2026) with only its top result and an `error` asking for one, and
about one request every 10 seconds (it answers faster ones with HTTP 429). The gazetteer uses only the top result.
Flash-flood locations go first, then the most recently alerted, so a run that stops early has placed the labels that
matter most.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator

from ..labels import alert_text, locations
from ..labels.coverage import latest_versions
from . import sources as S
from .client import Http, HttpError
from .sinks import Sink

log = logging.getLogger(__name__)

POLITE_DELAY = 10.5  # without a token OneMap allows about one search per 10 s (250 a minute with one)
BATCH = 10  # locations per landed file (about 4 minutes of searching), so a stopped run loses little


def _landed_files(sink: Sink, folder: str) -> Iterator[str]:
    for name in sink.list_names(folder):
        if name.startswith("."):  # a file still being written
            continue
        path = f"{folder}/{name}"
        if ".jsonl" in name:
            yield path
        elif "." not in name:  # archive/, live/, wayback/, date=.../
            yield from _landed_files(sink, path)


def landed_lines(sink: Sink, folder: str) -> Iterator[dict]:
    """Every JSON line landed under `folder`, gzipped or not."""
    for path in _landed_files(sink, folder):
        data = sink.read_bytes(path) or b""
        text = (gzip.decompress(data) if path.endswith(".gz") else data).decode("utf-8")
        yield from (json.loads(line) for line in text.splitlines() if line.strip())


def alert_locations(posts: Iterable[dict]) -> dict[str, dict]:
    """location_key -> {location, flood, last_posted} for every location a warning or flash-flood post names (the
    latest version of each post): its first spelling, whether a flash flood named it, and when it was last named."""
    found: dict[str, dict] = {}
    for post in latest_versions(posts).values():
        parsed = alert_text.parse(post.get("text"), service=bool(post.get("service")))
        if not parsed.is_alert:
            continue
        for place in parsed.locations:
            key = locations.location_key(place.location)
            if not key:
                continue
            entry = found.setdefault(
                key, {"location": place.location, "flood": False, "last_posted": ""}
            )
            entry["flood"] |= parsed.category == alert_text.FLASH_FLOOD
            entry["last_posted"] = max(
                entry["last_posted"], post.get("posted_at") or ""
            )
    return found


def accepted(query: str | None, body: object) -> bool:
    """Whether OneMap's top result places `query`: a point in Singapore whose road, name or building is the place the
    query names (locations.SAME_NAME). Its top result for "Kings Rd" is Kings Home on Joo Chiat Road; for
    "Bukit Timah Rd", an expressway gantry."""
    results = body.get("results") if isinstance(body, dict) else None
    try:
        top = results[0]
        lat, lon = float(top["LATITUDE"]), float(top["LONGITUDE"])
    except (TypeError, IndexError, KeyError, ValueError):
        return False
    if not (1.15 <= lat <= 1.48 and 103.59 <= lon <= 104.10):
        return False
    score = max(locations.name_match(query, top.get(f)) for f in NAME_FIELDS)
    return score >= locations.SAME_NAME


NAME_FIELDS = ("ROAD_NAME", "SEARCHVAL", "BUILDING")


def onemap_search(
    http: Http,
    sink: Sink,
    *,
    runner: str,
    polite_delay: float = POLITE_DELAY,
    deadline: float | None = None,
) -> dict:
    """Search every alert location OneMap hasn't placed yet, with the queries it hasn't been asked (so better
    `search_queries` retry the misses); land the answers under onemap_search/search/."""
    asked: dict[str, set[str]] = defaultdict(set)
    placed: set[str] = set()
    for line in landed_lines(sink, S.ONEMAP):
        key = locations.location_key(line.get("location"))
        asked[key].add(line.get("query"))
        if accepted(line.get("query"), line.get("body")):
            placed.add(key)
    todo = []
    for key, entry in alert_locations(landed_lines(sink, S.TELEGRAM)).items():
        queries = [
            (rank, query)
            for rank, query in enumerate(locations.search_queries(entry["location"]))
            if query not in asked.get(key, ())
        ]
        if key not in placed and queries:
            todo.append((key, entry, queries))
    # floods first, then the most recently alerted (two stable sorts)
    todo.sort(key=lambda item: item[1]["last_posted"], reverse=True)
    todo.sort(key=lambda item: not item[1]["flood"])
    log.info(
        "%d alert locations searched before (%d placed), %d to search",
        len(asked),
        len(placed),
        len(todo),
    )
    stats = {"locations": 0, "found": 0, "queries": 0, "failed": 0, "files": 0}
    batch: list[dict] = []

    def land() -> None:
        if batch:
            path = S.run_path(S.ONEMAP, "search", S.utcnow(), ext="jsonl.gz")
            sink.write_bytes(path, S.gzipped(S.jsonl(batch)))
            stats["files"] += 1
            batch.clear()

    for n, (key, entry, queries) in enumerate(todo):
        place = entry["location"]
        if deadline is not None and time.monotonic() >= deadline:
            stats["stopped_before"] = place
            break
        try:
            answers = list(
                _search(
                    http, place, key, queries, runner=runner, polite_delay=polite_delay
                )
            )
        except HttpError as exc:
            log.warning("%s: %s; it will be searched again next run", place, exc)
            stats["failed"] += 1
            continue
        batch += answers
        stats["locations"] += 1
        stats["queries"] += len(answers)
        stats["found"] += any(accepted(a["query"], a["body"]) for a in answers)
        if (n + 1) % BATCH == 0:
            land()
    land()
    return stats


def _search(
    http: Http,
    place: str,
    key: str,
    queries: list[tuple[int, str]],
    *,
    runner: str,
    polite_delay: float,
) -> Iterator[dict]:
    """OneMap's answers to `queries` ((rank, text), best first) for `place`, up to the first it accepts."""
    for rank, query in queries:
        resp = http.get(
            S.ONEMAP_SEARCH_URL,
            {
                "searchVal": query,
                "returnGeom": "Y",
                "getAddrDetails": "Y",
                "pageNum": "1",
            },
        )
        body = resp.json()
        yield S.record(
            S.ONEMAP,
            fetched_at=S.utcnow(),
            runner=runner,
            location=place,
            location_key=key,
            query=query,
            query_rank=rank,
            status=resp.status_code,
            body=body,
        )
        http.sleep(polite_delay)
        if accepted(query, body):
            return
