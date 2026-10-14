import gzip
import hashlib
import json
import re
from datetime import UTC, date, datetime, timedelta

from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest.sinks import LocalSink
from daisi_team_master_corgis_floodsense.ingest.wayback import (
    Capture,
    Recovered,
    list_captures,
    place_newest_pages,
    telegram_wayback,
)
from tests.fakes import FakeResponse, fake_http

SGFLOOD, PUB = S.TELEGRAM_ALIASES
START = datetime(2022, 1, 1, tzinfo=UTC)  # post i went up i hours after this


def page(channel: str, ids, *, newer: bool = True, title: str = "PUB Flood Alerts"):
    """A t.me/s/ page in Telegram's markup; with a single id it reads like an embedded post."""
    posts = "".join(
        f'<div class="tgme_widget_message_wrap js-widget_message_wrap">'
        f'<div class="tgme_widget_message js-widget_message" data-post="{channel}/{i}">'
        f'<a class="tgme_widget_message_owner_name" href="https://t.me/{channel}"><span dir="auto">{title}</span></a>'
        f'<div class="tgme_widget_message_text js-message_text" dir="auto">post {i}</div>'
        f'<time datetime="{(START + timedelta(hours=i)).isoformat()}" class="time">x</time>'
        f"</div></div>"
        for i in ids
    )
    more = (
        f'<a href="/s/{channel}?after={ids[-1]}" class="tme_messages_more js-messages_more" '
        f'data-after="{ids[-1]}"></a>'
        if newer and ids
        else ""
    )
    return f"<section>{posts}{more}</section>"


class FakeArchive:
    """web.archive.org: a CDX index over `captures` ({(timestamp, t.me URL): html}) and the captures themselves."""

    def __init__(self, captures: dict[tuple[str, str], str]):
        self.captures = captures
        self.reads: list[str] = []

    def routes(self) -> list:
        return [("web.archive.org/cdx", self.cdx), ("web.archive.org/web/", self.web)]

    def cdx(self, url, params):
        rows = [["timestamp", "original", "mimetype", "digest"]]
        for (timestamp, original), _ in sorted(self.captures.items()):
            listed = original.split("://", 1)[1].lower().startswith(params["url"])
            if listed and params["from"] <= timestamp[:8] <= params["to"]:
                digest = hashlib.sha1(f"{timestamp}{original}".encode()).hexdigest()
                rows.append([timestamp, original, "text/html", digest[:32].upper()])
        return FakeResponse(200, rows)

    def web(self, url, params):
        timestamp, original = re.fullmatch(
            r"https://web\.archive\.org/web/(\d{14})id_/(.+)", url
        ).groups()
        self.reads.append(f"{timestamp} {original}")
        return FakeResponse(200, raw=self.captures[(timestamp, original)].encode())


def run(archive: FakeArchive, sink: LocalSink, **kwargs) -> dict:
    http, *_ = fake_http(archive.routes())
    return telegram_wayback(http, sink, runner="test", polite_delay=0, **kwargs)


def landed(root) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((root / S.TELEGRAM / "wayback").iterdir())
        for line in gzip.decompress(path.read_bytes()).decode().splitlines()
    ]


def test_list_captures_keeps_the_readable_urls_inside_the_username_window():
    rows = {
        "t.me/s/sgflood": [
            ["20220801000000", "https://t.me/s/sgflood/150", "text/html", "A"],
            ["20220801000001", "https://t.me/s/SGflood/5", "text/html", "B"],
            ["20220801000002", "https://t.me/s/sgflood?before=61", "text/html", "C"],
            ["20220801000003", "https://t.me/s/sgflood", "text/html", "D"],
            ["20220801000004", "https://t.me/s/sgflood2/5", "text/html", "E"],
            ["20220801000005", "https://t.me/s/sgflood?q=rain", "text/html", "F"],
            ["20220801000006", "https://t.me/s/sgflood/151", "warc/revisit", "G"],
        ],
        "t.me/sgflood/": [
            [
                "20220801000007",
                "https://t.me/sgflood/140?embed=1&mode=tme",
                "text/html",
                "H",
            ],
            ["20220801000008", "https://t.me/sgflood/140", "text/html", "I"],  # no date
        ],
    }
    header = ["timestamp", "original", "mimetype", "digest"]
    http, session, _ = fake_http(
        [
            (
                "web.archive.org/cdx",
                lambda url, params: FakeResponse(
                    200, [header, *rows.get(params["url"], [])]
                ),
            )
        ]
    )
    captures = list_captures(http, SGFLOOD, until=date(2025, 3, 31), polite_delay=0)
    assert [
        (c.digest, c.kind, c.named, c.first, c.last)
        for c in captures
        if c.kind != "newest"
    ] == [
        ("A", "page", 150, 141, 159),
        ("B", "page", 5, 1, 14),
        ("C", "page", 61, 41, 60),
        ("H", "post", 140, 140, 140),
    ]
    assert [c.digest for c in captures if c.kind == "newest"] == ["D"]
    # sgflood is read only up to its rename, whatever --until says; both hosts, pages and posts
    assert sorted((c["params"]["url"], c["params"]["to"]) for c in session.calls) == [
        ("t.me/s/sgflood", "20231114"),
        ("t.me/sgflood/", "20231114"),
        ("telegram.me/s/sgflood", "20231114"),
        ("telegram.me/sgflood/", "20231114"),
    ]
    session.calls.clear()
    list_captures(http, PUB, until=date(2024, 11, 16), polite_delay=0)
    assert {(c["params"]["from"], c["params"]["to"]) for c in session.calls} == {
        ("20231114", "20241116")
    }


def test_a_newest_page_is_expected_between_the_posts_known_to_bracket_it():
    def capture(timestamp, kind, named=0):
        return Capture(
            timestamp, timestamp, timestamp, kind, named, named or 1, named or -1
        )

    early, _, late = place_newest_pages(
        [
            capture("20220101000000", "newest"),
            capture("20220102000000", "post", 40),
            capture("20220103000000", "newest"),
        ]
    )
    # the newest post on a page is at least the highest id named before it
    assert (early.named, late.named) == (0, 40)
    known = Recovered()
    assert list(known.expected(early)) == []
    assert known.expected(late) == range(22, 59)
    known.observe(
        datetime(2022, 1, 9, tzinfo=UTC),
        [
            {"message_id": 30, "posted_at": "2021-12-31T00:00:00+00:00"},
            {"message_id": 45, "posted_at": "2022-01-02T12:00:00+00:00"},
            {"message_id": 50, "posted_at": "2022-01-05T00:00:00+00:00"},
        ],
        page=False,
    )
    assert known.expected(early) == range(12, 45)  # newest at least 30, before 45
    assert known.expected(late) == range(27, 50)  # newest at least 45, before 50


def test_ids_passed_over_are_tried_again_once_more_is_known(tmp_path):
    archive = FakeArchive(
        {
            # the first capture: posts 1-9 were up; nothing named an id before it
            ("20220101093000", "https://t.me/s/sgflood"): page(
                "sgflood", range(1, 10), newer=False
            ),
            # by August, 6-94 had been deleted; post 5's time places the newest page
            ("20220801000000", "https://t.me/s/sgflood/100"): page(
                "sgflood", [5, *range(95, 106)]
            ),
        }
    )
    summary = run(archive, LocalSink(tmp_path))
    assert archive.reads == [
        "20220801000000 https://t.me/s/sgflood/100",
        "20220101093000 https://t.me/s/sgflood",
    ]
    assert summary["ids_recovered"] == 20
    assert summary["not_recovered"] == {"deleted_before_capture": "10-94"}


def test_each_post_is_read_from_as_few_captures_as_possible(tmp_path):
    archive = FakeArchive(
        {
            ("20220801000000", "https://t.me/s/sgflood/10"): page(
                "sgflood", range(1, 21)
            ),
            ("20220801000100", "https://t.me/s/sgflood/12"): page(
                "sgflood", range(2, 23)
            ),
            ("20220801000200", "https://t.me/s/sgflood/25"): page(
                "sgflood", range(15, 31), newer=False
            ),
            ("20220801000300", "https://telegram.me/s/sgflood/10"): page(
                "sgflood", range(1, 21)
            ),
            ("20220801000400", "https://t.me/sgflood/21?embed=1"): page(
                "sgflood", [21]
            ),
        }
    )
    sink = LocalSink(tmp_path)
    summary = run(archive, sink)
    assert archive.reads == [
        "20220801000000 https://t.me/s/sgflood/10",
        "20220801000200 https://t.me/s/sgflood/25",
    ]
    assert summary["ids_recovered"] == 30 and summary["not_recovered"] == {}
    assert summary["captures_listed"] == {"page": 4, "post": 1}
    first = landed(tmp_path)[0]
    assert {
        k: first[k]
        for k in ("source", "runner", "channel", "channel_title", "message_id")
    } == {
        "source": S.TELEGRAM,
        "runner": "test",
        "channel": "sgflood",
        "channel_title": "PUB Flood Alerts",
        "message_id": 1,
    }
    assert first["page_url"] == "https://t.me/s/sgflood/10"
    assert first["captured_at"] == "2022-08-01T00:00:00.000Z"
    assert (
        first["capture_url"]
        == "https://web.archive.org/web/20220801000000/https://t.me/s/sgflood/10"
    )
    assert first["posted_at"] == "2022-01-01T01:00:00+00:00"

    # a re-run knows what landed and reads nothing again
    again = run(archive, sink)
    assert len(archive.reads) == 2 and again["captures_read"] == 0
    assert again["ids_recovered"] == 30


def test_a_post_deleted_later_is_read_from_a_capture_taken_before(tmp_path):
    archive = FakeArchive(
        {
            # by Nov 2023 post 7 was gone; the Aug 2022 and Jan 2024 widgets are all that name it
            ("20231128000000", "https://t.me/s/pubfloodalerts/10"): page(
                "pubfloodalerts", [i for i in range(1, 21) if i != 7]
            ),
            ("20220801000000", "https://t.me/sgflood/7?embed=1"): page("sgflood", [7]),
            (
                "20240101000000",
                "https://t.me/pubfloodalerts/7?embed=1",
            ): "<p>Post not found</p>",
        }
    )
    summary = run(archive, LocalSink(tmp_path))
    assert archive.reads == [
        "20231128000000 https://t.me/s/pubfloodalerts/10",
        "20220801000000 https://t.me/sgflood/7?embed=1",
    ]
    assert summary["ids_recovered"] == 20 and summary["not_recovered"] == {}


def test_a_page_ending_at_the_newest_post_rules_out_newer_posts_for_a_while(tmp_path):
    archive = FakeArchive(
        {
            ("20220101203000", "https://t.me/s/sgflood/10"): page(
                "sgflood", range(1, 21), newer=False
            ),
            # ten minutes later the channel still ended at post 20: expected to show 21-27, it can't
            ("20220101204000", "https://telegram.me/s/sgflood/18"): page(
                "sgflood", range(8, 21), newer=False
            ),
            ("20220102160000", "https://t.me/sgflood/21?embed=1"): page(
                "sgflood", [21]
            ),
        }
    )
    run(archive, LocalSink(tmp_path))
    assert archive.reads == [
        "20220101203000 https://t.me/s/sgflood/10",
        "20220102160000 https://t.me/sgflood/21?embed=1",
    ]


def test_a_rerun_remembers_which_pages_ended_at_the_newest_post(tmp_path):
    archive = FakeArchive(
        {
            ("20220101203000", "https://t.me/s/sgflood/10"): page(
                "sgflood", range(1, 21), newer=False
            ),
            ("20220101204000", "https://telegram.me/s/sgflood/21"): page(
                "sgflood", range(8, 21), newer=False
            ),
        }
    )
    sink = LocalSink(tmp_path)
    run(archive, sink)
    run(archive, sink)
    assert archive.reads == ["20220101203000 https://t.me/s/sgflood/10"]
    assert {p["page_has_newer"] for p in landed(tmp_path)} == {False}


def test_what_a_page_shows_bounds_when_missing_posts_can_be_found():
    def at(hours):
        return START + timedelta(hours=hours)

    known = Recovered()
    # captured at 30h: posts 1-20 except 7 (deleted by then), and nothing newer than 20
    known.observe(
        at(30),
        [
            {"message_id": i, "posted_at": at(i).isoformat()}
            for i in range(1, 21)
            if i != 7
        ],
        page=True,
        has_newer=False,
    )
    assert known.can_add(7, at(29)) and not known.can_add(7, at(30))
    assert not known.can_add(21, at(30.5)) and known.can_add(21, at(31))
    # an embedded post says nothing about its neighbours, and has no link to newer posts
    known.observe(
        at(40),
        [{"message_id": 23, "posted_at": at(23).isoformat()}],
        page=False,
        has_newer=False,
    )
    assert known.can_add(22, at(40)) and not known.can_add(23, at(40))
    assert known.can_add(24, at(40.5))

    # post 24 went up after post 23 did, so nothing captured before 23h shows it
    fresh = Recovered()
    fresh.observe(
        at(40), [{"message_id": 23, "posted_at": at(23).isoformat()}], page=False
    )
    assert fresh.can_add(24, at(23.5)) and not fresh.can_add(24, at(22.5))


def test_captures_of_another_channel_are_left_out(tmp_path):
    archive = FakeArchive(
        {
            # after the rename sgflood belongs to someone else: outside its window, so never listed
            ("20240717000000", "https://t.me/s/sgflood/5"): page(
                "SGflood", range(1, 15), title="SHIP GRAVEYARD"
            ),
            # inside the window, a capture whose posts aren't PUB's is read but lands nothing
            ("20231101000000", "https://t.me/s/sgflood/5"): page(
                "sgflood", range(1, 15), title="Someone else"
            ),
            ("20231201000000", "https://t.me/s/pubfloodalerts/5"): page(
                "pubfloodalerts", range(1, 15), title="Gov.sg-PUB Flood Alerts"
            ),
        }
    )
    summary = run(archive, LocalSink(tmp_path))
    assert archive.reads == [
        "20231101000000 https://t.me/s/sgflood/5",
        "20231201000000 https://t.me/s/pubfloodalerts/5",
    ]
    assert summary["foreign_captures"] == 1 and summary["ids_recovered"] == 14
    assert {(p["channel"], p["channel_title"]) for p in landed(tmp_path)} == {
        ("pubfloodalerts", "Gov.sg-PUB Flood Alerts")
    }
