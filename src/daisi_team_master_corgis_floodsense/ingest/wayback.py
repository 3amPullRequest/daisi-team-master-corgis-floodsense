"""PUB's deleted Telegram posts, recovered from web archive (Wayback Machine) captures of the channel.

PUB has wiped its channel's history at least three times, each time keeping only a few posts: between February and
August 2022, between April and September 2024, and around 23 Mar 2025. So posts older than 19 Mar 2025 survive only
where web.archive.org captured them, under the username the channel had then (`sources.TELEGRAM_ALIASES`):

- channel pages: `t.me/s/<username>` (the newest ~20 posts) and `t.me/s/<username>/<id>` (~20 posts around <id>);
- embedded posts: `t.me/<username>/<id>?embed=1`.

Both parse like the live channel page. The archive holds thousands of captures, mostly repeats, and throttles busy
clients, so posts are recovered id by id from as few captures as possible. Each capture read lands as one file under
`pub_telegram_posts/wayback/`, which also marks it as read; a re-run rebuilds what it knows from those files.

How a run works:

1. List the captures. `list_captures` asks the archive's CDX index for every capture of the channel's pages and embedded
   posts under each username in its window, on both t.me and telegram.me. `_classify` turns each URL into a `Capture`
   and says which post ids it should show: a page around post n shows about n-9 to n+9, a `?before=n` page n-20 to
   n-1, an `?after=n` page n+1 to n+20, an embedded post just itself, and a "newest" page (t.me/s/<username>) the
   newest ~20 posts as of its time, which isn't known from the URL. Other URLs are dropped.
2. Keep score. `Recovered` holds what the captures read so far say about each post id: the posts recovered and when
   each was posted; the ids known to be gone (a channel page shows every post between its first and last, so an id
   missing inside it had been deleted by the time of that capture); and the capture times at which a page had no link
   to newer posts and so ended at the channel's newest post. From these, `expected` predicts which ids a newest page
   will show, `can_add` says whether a capture taken at a given time could still contribute a post (it isn't
   recovered, wasn't deleted by then, and had been posted by then), and `gain` counts how many ids a capture would add.
3. Walk the ids. `telegram_wayback` goes through post ids 1 up to the highest any capture names, and for each id not
   yet recovered reads the unread capture that covers it and would add the most missing ids (the earliest on a tie),
   until one shows the post or none can. Reading posts sharpens what the newest pages are expected to show, so the
   walk repeats until a whole pass reads nothing. It sleeps between requests, and stops at the deadline.
4. Read a capture. `_read` fetches the capture exactly as archived (the `id_` form of the URL, unrewritten), parses it
   with the same listing parser as the live page, discards a capture of another channel (the unrelated channel that
   later took the name sgflood; its title lacks "Flood Alerts"), and lands the posts with `captured_at`, so silver
   dates a post's version by when the archive saw it. The file is written even when empty, since its existence is what
   marks the capture read.

The summary says what was recovered by year and what wasn't, with the ids by reason: deleted before any capture saw
them, covered by no capture at all, or covered by captures that were read and didn't show them.
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from functools import cached_property
from urllib.parse import parse_qsl

from . import sources as S
from .client import Http, HttpError
from .scrapers import parse_telegram_listing
from .sinks import Sink

log = logging.getLogger(__name__)

ARCHIVE = "https://web.archive.org/web"
CDX_URL = "https://web.archive.org/cdx/search/cdx"
# the archive keeps separate captures under each host
HOSTS = ("t.me", "telegram.me")
# seconds between archive requests; it throttles clients making more than ~15 a minute
POLITE_DELAY = 4.0
# a page around post <id> shows about 10 posts on either side of it; a newest page, the newest ~20
REACH = 9
# once a page ends at the channel's newest post, expect nothing newer for this long
NEWEST_GAP = timedelta(hours=1)
# in every title PUB's channel has had, and not in the later sgflood's
TITLE_MARK = "Flood Alerts"
_NEVER = datetime.max.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Capture:
    """One archived copy of a t.me URL. `first` to `last` are the post ids it is expected to show; for a newest
    page they depend on what has been recovered (`Recovered.expected`)."""

    timestamp: str  # 14 digits, UTC, as the archive names captures
    url: str  # the t.me URL captured
    digest: str  # the archive's hash of the captured bytes
    kind: str  # "page" (around a post id), "newest" (t.me/s/<username>) or "post"
    # the post id in the URL, which existed when captured; for a newest page, the highest
    # id any earlier capture named
    named: int
    first: int
    last: int

    @cached_property
    def at(self) -> datetime:
        return datetime.strptime(self.timestamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)

    @property
    def path(self) -> str:
        return S.wayback_path(self.timestamp, self.digest)


def list_captures(
    http: Http,
    alias: S.TelegramAlias,
    *,
    until: date,
    polite_delay: float = POLITE_DELAY,
) -> list[Capture]:
    """Captures of the alias's channel pages and embedded posts taken inside its window, up to `until`."""
    last = min(alias.last or until, until)
    captures: list[Capture] = []
    if last < alias.first:
        return captures
    for host in HOSTS:
        for prefix in (f"{host}/s/{alias.username}", f"{host}/{alias.username}/"):
            query = {
                "url": prefix,
                "matchType": "prefix",
                "from": f"{alias.first:%Y%m%d}",
                "to": f"{last:%Y%m%d}",
                "filter": "statuscode:200",
                "fl": "timestamp,original,mimetype,digest",
                "output": "json",
            }
            rows = http.get(CDX_URL, query).json()
            http.sleep(polite_delay)
            # the first row names the fields
            for timestamp, url, mimetype, digest in rows[1:]:
                capture = _classify(alias.username, timestamp, url, digest)
                if capture and mimetype == "text/html":
                    captures.append(capture)
    return captures


_TME_URL = re.compile(
    r"https?://(?:www\.)?(?:t|telegram)\.me(?::\d+)?/(?P<path>[^?#]*)(?:\?(?P<query>[^#]*))?",
    re.IGNORECASE,
)


def _classify(username: str, timestamp: str, url: str, digest: str) -> Capture | None:
    """What a captured URL is expected to show. The prefix search also returns URLs with no post to read (the
    channel's info page, a post page without the embed) and longer usernames; those give None."""
    match = _TME_URL.fullmatch(url)
    if not match:
        return None
    parts = match["path"].lower().rstrip("/").split("/")
    query = dict(parse_qsl(match["query"] or ""))
    if len(parts) == 3 and parts[:2] == ["s", username] and parts[2].isdigit():
        n = int(parts[2])
        kind, named, first, last = "page", n, n - REACH, n + REACH
    elif parts == ["s", username] and query.get("before", "").isdigit():
        n = int(query["before"])
        kind, named, first, last = "page", n, n - 20, n - 1
    elif parts == ["s", username] and query.get("after", "").isdigit():
        n = int(query["after"])
        kind, named, first, last = "page", n, n + 1, n + 20
    elif parts == ["s", username] and not query:
        kind, named, first, last = "newest", 0, 0, -1
    elif (
        len(parts) == 2
        and parts[0] == username
        and parts[1].isdigit()
        and query.get("embed") == "1"
    ):
        n = int(parts[1])
        kind, named, first, last = "post", n, n, n
    else:
        return None
    return Capture(timestamp, url, digest, kind, named, max(first, 1), last)


def place_newest_pages(captures: list[Capture]) -> list[Capture]:
    """Give each newest page (t.me/s/<username>) the highest post id any earlier capture named: the newest post it
    shows is at least that one. Post ids are shared by all the usernames."""
    placed, highest = [], 0
    for capture in sorted(captures, key=lambda c: (c.timestamp, c.kind == "newest")):
        if capture.kind == "newest":
            capture = replace(capture, named=highest)
        highest = max(highest, capture.named)
        placed.append(capture)
    return placed


class Recovered:
    """What the captures read so far say about each post id."""

    def __init__(self) -> None:
        self.read: set[str] = set()  # landed paths of the captures read
        self.posted_at: dict[int, datetime | None] = {}  # every post recovered
        # recovered ids with a posted_at, ascending, and those times (ids go up with time)
        self._dated: list[int] = []
        self._times: list[datetime] = []
        self.gone: dict[int, datetime] = {}  # id -> first capture known to lack it
        # (capture time, the newest post id then), from pages with no link to newer posts
        self._newest: list[tuple[datetime, int]] = []

    @classmethod
    def from_landed(cls, sink: Sink) -> Recovered:
        """What earlier runs landed."""
        known = cls()
        folder = f"{S.TELEGRAM}/wayback"
        for name in sink.list_names(folder):
            path = f"{folder}/{name}"
            known.read.add(path)
            data = gzip.decompress(sink.read_bytes(path) or b"").decode()
            posts = [json.loads(line) for line in data.splitlines()]
            if posts:
                known.observe(
                    datetime.fromisoformat(posts[0]["captured_at"]),
                    posts,
                    page="/s/" in posts[0]["page_url"].lower(),
                    has_newer=posts[0].get("page_has_newer", True),
                )
        return known

    def observe(
        self, at: datetime, posts: list[dict], *, page: bool, has_newer: bool = True
    ) -> None:
        """Take in the posts of one capture. A channel page shows every post between its first and last, so an id
        missing in between had been deleted by then; and a page without a link to newer posts ends at the newest."""
        for post in posts:
            k = post["message_id"]
            if k in self.posted_at:
                continue
            self.posted_at[k] = _posted_at(post)
            if self.posted_at[k]:
                i = bisect.bisect_left(self._dated, k)
                self._dated.insert(i, k)
                self._times.insert(i, self.posted_at[k])
        ids = {post["message_id"] for post in posts}
        if not page or not ids:
            return
        for k in set(range(min(ids), max(ids))) - ids:
            if at < self.gone.get(k, _NEVER):
                self.gone[k] = at
        if not has_newer:
            self._newest.append((at, max(ids)))

    def expected(self, capture: Capture) -> range:
        """The post ids a capture is expected to show. A newest page shows the ~20 newest posts of its time; the
        newest of them is at least the highest id known to be up by then (named by an earlier capture, or recovered
        and posted earlier), and comes before the first post recovered that went up later."""
        if capture.kind != "newest":
            return range(capture.first, capture.last + 1)
        i = bisect.bisect_right(self._times, capture.at)
        floor = max(capture.named, self._dated[i - 1] if i else 0)
        if not floor:
            return range(0)
        ceiling = floor + 2 * REACH
        if i < len(self._dated):
            ceiling = max(min(ceiling, self._dated[i] - 1), floor)
        return range(max(floor - 2 * REACH, 1), ceiling + 1)

    def can_add(self, k: int, at: datetime) -> bool:
        """Whether a capture taken at `at` could add post k: it isn't recovered yet, wasn't deleted by then, and, as
        far as is known, had been posted."""
        if k in self.posted_at or at >= self.gone.get(k, _NEVER):
            return False
        i = bisect.bisect_left(self._dated, k)
        if i and at < self._times[i - 1]:
            return False  # an earlier post wasn't up yet
        return all(
            at >= seen + NEWEST_GAP for seen, newest in self._newest if newest < k
        )

    def gain(self, capture: Capture) -> int:
        return sum(self.can_add(k, capture.at) for k in self.expected(capture))


def telegram_wayback(
    http: Http,
    sink: Sink,
    *,
    runner: str,
    until: date = S.WAYBACK_UNTIL,
    polite_delay: float = POLITE_DELAY,
    deadline: float | None = None,
) -> dict:
    """Recover the channel's posts from captures taken up to `until`. For each post id still missing, read the
    capture expected to add the most missing ids (the earliest, on a tie) until one shows it or none can. What a
    newest page is expected to show sharpens as posts are recovered, so the ids are walked again until a walk reads
    nothing."""
    captures = place_newest_pages(
        [
            capture
            for alias in S.TELEGRAM_ALIASES
            for capture in list_captures(
                http, alias, until=until, polite_delay=polite_delay
            )
        ]
    )
    known = Recovered.from_landed(sink)
    newest = [c for c in captures if c.kind == "newest"]
    covering: dict[int, list[Capture]] = defaultdict(list)
    for capture in captures:
        for k in range(capture.first, capture.last + 1):  # none for a newest page
            covering[k].append(capture)
    top = max((c.named for c in captures), default=0)
    log.info(
        "%d captures name post ids up to %d; %d read before",
        len(captures),
        top,
        len(known.read),
    )
    stats: Counter[str] = Counter()
    while True:
        read_before = stats["captures_read"]
        for k in range(1, top + 1):
            while k not in known.posted_at:
                candidates = covering[k] + [c for c in newest if k in known.expected(c)]
                options = [
                    c
                    for c in candidates
                    if c.path not in known.read and known.can_add(k, c.at)
                ]
                if not options:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    log.info("max runtime reached; stopping before post %d", k)
                    summary = _summary(captures, covering, newest, known, stats, top)
                    return {**summary, "stopped_before": k}
                best = min(options, key=lambda c: (-known.gain(c), c.timestamp))
                stats.update(_read(http, sink, best, known, runner=runner))
                http.sleep(polite_delay)
        if stats["captures_read"] == read_before:
            return _summary(captures, covering, newest, known, stats, top)


def _read(
    http: Http, sink: Sink, capture: Capture, known: Recovered, *, runner: str
) -> Counter[str]:
    """Fetch one capture as it was taken, land its posts and take them in. A capture of another channel (the later
    sgflood) lands empty."""
    fetched_at = S.utcnow()
    try:
        resp = http.get(f"{ARCHIVE}/{capture.timestamp}id_/{capture.url}")
        html = resp.content.decode("utf-8", "replace")
        posts, has_newer = parse_telegram_listing(html)
    except HttpError as exc:
        if exc.status not in (403, 404, 410):
            raise
        log.warning("%s %s: HTTP %d", capture.timestamp, capture.url, exc.status)
        posts, has_newer = [], True
    titles = {p["channel_title"] for p in posts if p.get("channel_title")}
    others = sorted(t for t in titles if TITLE_MARK not in t)
    if others:
        log.warning("%s %s is %s's; skipped", capture.timestamp, capture.url, others)
        posts = []
    new = sum(p["message_id"] not in known.posted_at for p in posts)
    page = capture.kind != "post"
    # a channel page without a link to newer posts ended at the newest post then
    newer = {"page_has_newer": has_newer} if page else {}
    lines = [
        S.record(
            S.TELEGRAM,
            fetched_at=fetched_at,
            runner=runner,
            page_url=capture.url,
            captured_at=S.iso(capture.at),
            capture_url=f"{ARCHIVE}/{capture.timestamp}/{capture.url}",
            **newer,
            **post,
        )
        for post in posts
    ]
    # written even when empty: it marks the capture read
    sink.write_bytes(capture.path, S.gzipped(S.jsonl(lines)))
    known.read.add(capture.path)
    known.observe(capture.at, posts, page=page, has_newer=has_newer)
    log.info("%s %s: %d posts, %d new", capture.timestamp, capture.url, len(posts), new)
    return Counter(
        captures_read=1,
        posts_landed=len(posts),
        new_ids=new,
        foreign_captures=int(bool(others)),
    )


def _summary(
    captures: list[Capture],
    covering: dict[int, list[Capture]],
    newest: list[Capture],
    known: Recovered,
    stats: Counter[str],
    top: int,
) -> dict:
    missing: dict[str, list[int]] = defaultdict(list)
    for k in range(1, top + 1):
        if k in known.posted_at:
            continue
        if k in known.gone:
            # a later page skips it and no earlier capture shows it (or it's in an album)
            missing["deleted_before_capture"].append(k)
        elif not covering[k] and not any(k in known.expected(c) for c in newest):
            missing["never_captured"].append(k)
        else:
            missing["not_found"].append(k)
    years = Counter(t.year for t in known.posted_at.values() if t)
    return {
        "captures_listed": dict(Counter(c.kind for c in captures)),
        "captures_read": stats["captures_read"],
        "posts_landed": stats["posts_landed"],
        "new_ids": stats["new_ids"],
        "foreign_captures": stats["foreign_captures"],
        "ids_recovered": len(known.posted_at),  # by this run and earlier ones
        "recovered_by_year": dict(sorted(years.items())),
        "not_recovered": {reason: _ranges(ids) for reason, ids in missing.items()},
    }


def _posted_at(post: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(post["posted_at"])
    except (TypeError, ValueError):
        return None


def _ranges(ids: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3,7'"""
    spans: list[list[int]] = []
    for k in sorted(ids):
        if spans and k == spans[-1][1] + 1:
            spans[-1][1] = k
        else:
            spans.append([k, k])
    return ",".join(f"{a}-{b}" if a < b else str(a) for a, b in spans)
