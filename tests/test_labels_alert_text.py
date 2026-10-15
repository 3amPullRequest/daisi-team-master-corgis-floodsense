"""PUB alert text, read from real posts (tests/fixtures/pub_telegram_texts.jsonl, one per format and every edge case
found in the archive)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from daisi_team_master_corgis_floodsense.labels import alert_text as T
from daisi_team_master_corgis_floodsense.ml import contract as C

FIXTURES = Path(__file__).parent / "fixtures"
POSTS = {
    post["message_id"]: post
    for post in map(
        json.loads,
        (FIXTURES / "pub_telegram_texts.jsonl").read_text("utf-8").splitlines(),
    )
}


def parse(message_id: int) -> T.Parsed:
    post = POSTS[message_id]
    return T.parse(post["text"], service=post["service"])


def places(message_id: int) -> list[tuple[str, str | None]]:
    return [(loc.location, loc.stamp) for loc in parse(message_id).locations]


@pytest.mark.parametrize(
    ("message_id", "category", "pattern"),
    [
        (31, C.WARNING, "warning_rising_waters"),
        (72, C.WARNING, "warning_wave"),
        (476, C.WARNING, "warning_caution"),
        (635, C.WARNING, "warning_bracket"),
        (1374, C.WARNING, "warning_bracket"),  # "[Risk of Flash Flood]", singular
        (113, C.FLASH_FLOOD, "flood_siren"),
        (1013, C.FLASH_FLOOD, "flood_bracket"),
        (98, T.FLOOD_SUBSIDED, "subsided_arrow"),
        (1197, T.FLOOD_SUBSIDED, "subsided_plain"),
        (68, T.HEAVY_RAIN, "heavy_rain"),
        (
            66,
            T.HEAVY_RAIN,
            "heavy_rain",
        ),  # with a "Heavy Rain Warning" headline above it
        (627, T.HEAVY_RAIN_EASED, "heavy_rain_eased"),
        (1, T.NOT_RELEVANT, "service"),
        (1074, T.NOT_RELEVANT, "empty"),  # an album part without text
    ],
)
def test_every_format_is_recognised(message_id, category, pattern):
    parsed = parse(message_id)
    assert (parsed.category, parsed.pattern) == (category, pattern)


def test_posts_that_only_mention_floods_are_unmatched_not_alerts():
    obu = parse(
        1269
    )  # "Flash flood alerts will be available on the On-Board Unit (OBU) ..."
    assert (obu.category, obu.hint, obu.locations) == (T.UNMATCHED, C.FLASH_FLOOD, ())
    assert not obu.is_alert
    scam = parse(1264)  # ScamShield advisory
    assert (scam.category, scam.hint) == (T.UNMATCHED, None)


def test_list_formats_give_one_location_per_bullet():
    assert places(31) == [
        ("Siang Kuang Ave", "11:00"),
        ("Puay Hee Ave / Siak Kew Ave", "11:04"),
        ("Upper Paya Lebar Rd", "11:10"),
        ("Happy Ave North", "11:14"),
    ]
    assert places(72) == [
        ("Siang Kuang Ave", "18:58"),
        ("Puay Hee Ave / Siak Kew Ave", "19:03"),
        ("Lor 2 Toa Payoh Blk 122", "19:15"),
    ]
    assert places(557) == [
        ("Harper Rd", "10:10"),
        ("Ubi Ave 1", "10:10"),
        ("Tai Seng Dr", "10:08"),
    ]
    assert places(303) == [
        ("TPE Slip Rd (Punggol West Flyover)", "15:32")
    ]  # "•" bullets, hashtag after


def test_single_location_formats():
    assert places(635) == [("Jalan Boonlay (Bus stop B21339)", "17:41")]
    assert places(1630) == [
        ("Traffic Junction of Pesawat Dr and Jurong Port road", "14:13")
    ]  # "[14:13hours]"
    assert places(1374) == [
        ("Riverside Rd (near Admiralty Rd)", "16:10")
    ]  # full stop before the stamp
    assert places(1217) == [
        ("Junction of Toa Payoh Lor 1 and Toa Payoh Lor 2", "15:52")
    ]  # stamp on the next line
    assert places(1013) == [
        ("Wan Tho Ave (Pheng Geck Ave)", "15:10")
    ]  # "Issued 1510 hours."
    assert places(1299) == [
        ("Jurong Town Hall Road (towards PIE) before Jurong East Street 11", "07:49")
    ]


def test_drifted_and_broken_layouts():
    # a bullet left out
    assert places(424) == [("KPE (towards ECP) near Buangkok East Drive", "10:11")]
    # a doubled stamp: "[12:53 hours [12:59 hours]" keeps the first
    assert ("Harvey Rd / Macpherson Rd", "12:53") in places(89)
    assert len(places(89)) == 7
    # one sensor's roads, joined with ";", stay one alert
    assert len(places(959)) == 1 and places(959)[0][0].startswith(
        "Upp Paya Lebar Service Rd"
    )


def test_alerts_without_stamps_are_flagged():
    flood = parse(96)
    assert flood.problem == T.NO_STAMP
    assert [loc.location for loc in flood.locations] == [
        "Junction of Ubi Ave 3 and Eunos Link",
        "PIE slip road entrance / exit at Bedok North Ave 3 exit",
    ]
    assert parse(1268).problem == T.NO_STAMP and parse(1268).is_alert


def test_subsided_posts():
    assert places(98) == [("Bedok North Ave 3", "14:25")]
    assert places(133) == [
        ("Jurong East St 32", "15:50")
    ]  # "... has subsided. Roads are now passable. [..]"
    assert places(1364) == [
        ("Junction of Boon Lay Way and Corporation Road", "14:42")
    ]  # "[14:42 hours]"
    assert places(1626) == [
        ("Riverside Road", "15:35")
    ]  # "Issued 1535 hours", no full stop


def test_heavy_rain_regions_window_and_stamp():
    def read(message_id):
        parsed = parse(message_id)
        return parsed.regions, parsed.valid, parsed.issued

    assert read(68) == (("northern", "western", "central"), ("16:05", "17:45"), "15:40")
    assert read(626) == (
        ("southern", "eastern", "central"),
        ("12:10", "13:00"),
        "11:52",
    )
    assert read(66) == (("many",), ("20:05", "20:55"), "19:56")
    assert read(921)[2] == "13:50"  # "[Issued by NEA, [13:50 hours]"
    assert read(1028)[2] == "05:28"  # "[Issued by NEA, 05:28 hours hours]"
    assert parse(682).issued == "23:20"  # eased, "[Issued by NEA, 23: 20 hours]"
    assert parse(1206).issued == "16:45"  # eased, "[Issued by NEA, 16:45]"


def test_stamps_are_moments_on_the_day_of_posting_or_the_day_before():
    row = T.parse_row(POSTS[1630]["text"], POSTS[1630]["posted_at"])
    assert datetime.fromisoformat(row["locations"][0]["stamp_ts"]) == datetime(
        2026, 10, 1, 14, 13, tzinfo=T.SGT
    )
    # posted 00:05 on 23 Mar, issued "23:49": the night before
    row = T.parse_row(POSTS[1122]["text"], POSTS[1122]["posted_at"])
    assert datetime.fromisoformat(row["issued_ts"]) == datetime(
        2025, 3, 22, 23, 49, tzinfo=T.SGT
    )
    assert datetime.fromisoformat(row["valid_from"]) == datetime(
        2025, 3, 23, 0, 15, tzinfo=T.SGT
    )
    assert datetime.fromisoformat(row["valid_to"]) == datetime(
        2025, 3, 23, 1, 45, tzinfo=T.SGT
    )


def test_a_window_ending_at_24_00_ends_at_midnight():
    row = T.parse_row(POSTS[323]["text"], POSTS[323]["posted_at"])
    assert datetime.fromisoformat(row["valid_to"]) == datetime(
        2022, 10, 31, 0, 0, tzinfo=T.SGT
    )


def test_a_stamp_a_few_minutes_after_posting_is_clock_skew_not_yesterday():
    posted = datetime(2025, 5, 1, 15, 50, tzinfo=T.SGT)
    assert T.stamp_time("15:52", posted) == datetime(2025, 5, 1, 15, 52, tzinfo=T.SGT)
    assert T.stamp_time("23:58", datetime(2025, 5, 2, 0, 3, tzinfo=T.SGT)).day == 1


def test_api_descriptions_read_like_posts():
    flood = T.parse(
        "[FLASH FLOOD OCCURRED]  \n\nFlash flood at Boon Lay Avenue. Please avoid the area. Issued 1427 hours."
    )
    assert (flood.category, places_of(flood)) == (
        C.FLASH_FLOOD,
        [("Boon Lay Avenue", "14:27")],
    )
    subsided = T.parse("Flash flood subsided at Boon Lay Avenue. Issued 1454 hours.")
    assert (subsided.category, places_of(subsided)) == (
        T.FLOOD_SUBSIDED,
        [("Boon Lay Avenue", "14:54")],
    )


def places_of(parsed: T.Parsed) -> list[tuple[str, str | None]]:
    return [(loc.location, loc.stamp) for loc in parsed.locations]


def test_parse_row_matches_its_spark_schema():
    row = T.parse_row(POSTS[72]["text"], POSTS[72]["posted_at"])
    top = [field.split(" ")[0] for field in _top_level_fields(T.ROW_SCHEMA)]
    assert list(row) == top
    assert set(row["locations"][0]) == {"location", "stamp", "stamp_ts"}


def _top_level_fields(ddl: str) -> list[str]:
    fields, depth, current = [], 0, ""
    for char in ddl:
        depth += char == "<"
        depth -= char == ">"
        if char == "," and depth == 0:
            fields.append(current.strip())
            current = ""
        else:
            current += char
    return [*fields, current.strip()]


def test_every_category_is_known_and_alert_kinds_match_the_contract():
    assert set(C.ALERT_KINDS) <= set(T.CATEGORIES)
    assert {fmt.category for fmt in T.FORMATS} <= set(T.CATEGORIES)
    assert len({fmt.pattern for fmt in T.FORMATS}) == len(T.FORMATS)
