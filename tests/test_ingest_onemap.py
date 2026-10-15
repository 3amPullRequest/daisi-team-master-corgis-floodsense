import gzip
import json

from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest.onemap import (
    landed_lines,
    onemap_search,
)
from daisi_team_master_corgis_floodsense.ingest.sinks import LocalSink
from tests.fakes import FakeResponse, fake_http

WARNING = (
    "[Risk of Flash Floods]\n\nDue to heavy rain, please avoid this location for the next 1 hour: "
    "TPE Slip Rd (Punggol West Flyover) [15:32 hours]"
)
FLOOD = "[FLASH FLOOD OCCURRED]\n\nFlash flood at Riverside Road. Please avoid the area. Issued 1513 hours."
SCAM = "Government officials will NEVER ask you to transfer money or disclose bank log-in details over a phone call."


def land_posts(sink, texts):
    lines = [
        S.record(
            S.TELEGRAM,
            fetched_at=S.utcnow(),
            runner="test",
            message_id=i,
            posted_at="2026-09-30T07:15:39+00:00",
            text=text,
            service=False,
        )
        for i, text in enumerate(texts, start=1)
    ]
    sink.write_bytes(
        f"{S.TELEGRAM}/archive/date=2026-10-13/posts.jsonl.gz",
        S.gzipped(S.jsonl(lines)),
    )


def onemap(found: dict[str, tuple[float, float]]):
    """OneMap's search: the top result for a query in `found`, else no results."""

    def answer(url, params):
        point = found.get(params["searchVal"])
        results = (
            [
                {
                    "SEARCHVAL": params["searchVal"].upper(),
                    "ADDRESS": "x",
                    "LATITUDE": str(point[0]),
                    "LONGITUDE": str(point[1]),
                }
            ]
            if point
            else []
        )
        return FakeResponse(
            200,
            {
                "found": len(results),
                "results": results,
                "error": "Authentication token missing",
            },
        )

    return [("onemap.gov.sg", answer)]


def test_each_alert_location_is_searched_until_a_query_finds_it(tmp_path):
    sink = LocalSink(tmp_path)
    land_posts(sink, [WARNING, FLOOD, SCAM])
    http, session, _ = fake_http(
        onemap(
            {"Punggol West Flyover": (1.40, 103.90), "Riverside Road": (1.44, 103.80)}
        )
    )

    summary = onemap_search(http, sink, runner="test", polite_delay=0)

    asked = [call["params"]["searchVal"] for call in session.calls]
    # full text, then without brackets, then the bracket's place, which OneMap knows; the scam post isn't searched
    assert asked == [
        "Riverside Road",
        "TPE Slip Rd (Punggol West Flyover)",
        "TPE Slip Rd",
        "Punggol West Flyover",
    ]
    assert summary == {
        "locations": 2,
        "found": 2,
        "queries": 4,
        "failed": 0,
        "files": 1,
    }
    lines = list(landed_lines(sink, S.ONEMAP))
    assert [line["query_rank"] for line in lines] == [0, 0, 1, 2]
    assert {line["location"] for line in lines} == {
        "Riverside Road",
        "TPE Slip Rd (Punggol West Flyover)",
    }


def test_a_rerun_asks_only_about_new_locations(tmp_path):
    sink = LocalSink(tmp_path)
    land_posts(sink, [FLOOD])
    http, session, _ = fake_http(onemap({}))
    first = onemap_search(http, sink, runner="test", polite_delay=0)
    assert (
        first["found"] == 0 and first["locations"] == 1
    )  # not found still lands, so it isn't asked again

    again = onemap_search(http, sink, runner="test", polite_delay=0)
    assert again["locations"] == 0 and len(session.calls) == 1


def test_landed_lines_skip_files_still_being_written(tmp_path):
    sink = LocalSink(tmp_path)
    (tmp_path / S.ONEMAP / "search").mkdir(parents=True)
    (tmp_path / S.ONEMAP / "search" / ".partial.jsonl.gz.1.tmp").write_bytes(
        gzip.compress(b"{not json")
    )
    (tmp_path / S.ONEMAP / "search" / "a.jsonl").write_text(
        json.dumps({"location": "x"}) + "\n"
    )
    assert list(landed_lines(sink, S.ONEMAP)) == [{"location": "x"}]


def test_flood_locations_go_first_then_the_most_recent(tmp_path):
    sink = LocalSink(tmp_path)
    old, new = (
        WARNING.replace("TPE Slip Rd (Punggol West Flyover)", place)
        for place in ("Harper Rd", "Ubi Ave 1")
    )
    lines = [
        S.record(S.TELEGRAM, fetched_at=S.utcnow(), runner="test", message_id=i,
                 posted_at=posted, text=text, service=False)
        for i, (posted, text) in enumerate(
            [("2023-06-27T02:13:08+00:00", old), ("2026-10-01T06:20:39+00:00", new),
             ("2024-11-22T07:20:23+00:00", FLOOD)], start=1)
    ]  # fmt: skip
    sink.write_bytes(f"{S.TELEGRAM}/live/date=2026-10-01/x.jsonl", S.jsonl(lines))
    http, session, _ = fake_http(
        onemap(
            {
                "Harper Rd": (1.33, 103.89),
                "Ubi Ave 1": (1.33, 103.9),
                "Riverside Road": (1.44, 103.8),
            }
        )
    )
    onemap_search(http, sink, runner="test", polite_delay=0)
    asked = [call["params"]["searchVal"] for call in session.calls]
    assert asked == ["Riverside Road", "Ubi Ave 1", "Harper Rd"]


def test_an_answer_must_name_the_place_searched():
    from daisi_team_master_corgis_floodsense.ingest.onemap import accepted

    def body(road, searchval, building="NIL", lat=1.316, lon=103.899):
        top = {"ROAD_NAME": road, "SEARCHVAL": searchval, "BUILDING": building, "LATITUDE": str(lat),
               "LONGITUDE": str(lon)}  # fmt: skip
        return {"found": 1, "results": [top]}

    # OneMap's real top answers (14 Oct 2026)
    assert not accepted(
        "Kings Rd",
        body(
            "JOO CHIAT ROAD",
            "KINGS HOME JOO CHIAT COMPLEX",
            "KINGS HOME JOO CHIAT COMPLEX",
        ),
    )
    erp = "BT TIMAH EXPRESSWAY (SB BETW DAIRY FARM RD & PIE) - ERP(54)"
    assert not accepted("Bukit Timah Rd", body("NIL", erp, erp))
    assert accepted("Dunearn Rd", body("DUNEARN ROAD", "DUNEARN ROAD"))
    assert accepted(
        "Sime Darby Centre",
        body("DUNEARN ROAD", "SIME DARBY CENTRE", "SIME DARBY CENTRE"),
    )
    assert not accepted(
        "Dunearn Rd", body("DUNEARN ROAD", "DUNEARN ROAD", lat=3.1)
    )  # not in Singapore
    assert not accepted("Dunearn Rd", {"found": 0, "results": []})


def test_an_unplaced_location_is_asked_only_new_queries(tmp_path):
    sink = LocalSink(tmp_path)
    land_posts(sink, [WARNING])  # TPE Slip Rd (Punggol West Flyover)
    earlier = [
        S.record(S.ONEMAP, fetched_at=S.utcnow(), runner="test", location="TPE Slip Rd (Punggol West Flyover)",
                 query=query, query_rank=rank, status=200, body={"found": 0, "results": []})
        for rank, query in enumerate(["TPE Slip Rd (Punggol West Flyover)", "TPE Slip Rd"])
    ]  # fmt: skip
    sink.write_bytes(f"{S.ONEMAP}/search/date=2026-10-14/a.jsonl", S.jsonl(earlier))
    http, session, _ = fake_http(onemap({"Punggol West Flyover": (1.40, 103.90)}))

    onemap_search(http, sink, runner="test", polite_delay=0)
    assert [call["params"]["searchVal"] for call in session.calls] == [
        "Punggol West Flyover"
    ]
    # now placed, so a third run asks nothing
    onemap_search(http, sink, runner="test", polite_delay=0)
    assert len(session.calls) == 1
