import gzip
import json
import re
from datetime import date
from pathlib import Path

from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest.history import (
    api_days,
    api_start,
    hist_csv,
    telegram_archive,
)
from daisi_team_master_corgis_floodsense.ingest.sinks import LocalSink
from tests.fakes import FakeResponse, api_page, fake_http, no_data

FIXTURES = Path(__file__).parent / "fixtures"
RAINFALL = S.API_SOURCES_BY_NAME["nea_rainfall"]
RADAR = S.API_SOURCES_BY_NAME["nea_radar_70km"]


def gz_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in gzip.decompress(path.read_bytes()).decode().splitlines()
    ]


def test_api_days_lands_each_day_once(tmp_path):
    sink = LocalSink(tmp_path)
    http, session, _ = fake_http(
        [("rainfall", [api_page("readings", 25, token="p2"), api_page("readings", 12)])]
    )
    summary = api_days(
        http,
        sink,
        sources=[RAINFALL],
        runner="test",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
    )
    assert summary["nea_rainfall"]["landed"] == 1
    pages = gz_lines(tmp_path / S.backfill_path("nea_rainfall", date(2025, 1, 1)))
    assert [p["request"]["page"] for p in pages] == [1, 2]
    assert pages[1]["request"]["params"] == {
        "date": "2025-01-01",
        "paginationToken": "p2",
    }

    again = api_days(
        http,
        sink,
        sources=[RAINFALL],
        runner="test",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
    )
    assert again["nea_rainfall"]["skipped"] == 1 and len(session.calls) == 2


def test_api_days_marks_empty_days(tmp_path):
    http, *_ = fake_http([("weather-radar-images", no_data())])
    summary = api_days(
        http,
        LocalSink(tmp_path),
        sources=[RADAR],
        runner="test",
        start=date(2026, 7, 2),
        end=date(2026, 7, 2),
    )
    assert summary["nea_radar_70km"]["empty"] == 1
    (page,) = gz_lines(tmp_path / S.backfill_path("nea_radar_70km", date(2026, 7, 2)))
    assert page["status"] == 404 and page["body"]["code"] == 17


def test_api_days_stops_at_the_deadline(tmp_path):
    http, session, _ = fake_http([])
    summary = api_days(
        http,
        LocalSink(tmp_path),
        sources=[RAINFALL],
        runner="test",
        start=date(2025, 1, 1),
        end=date(2025, 1, 5),
        deadline=0,
    )
    assert (
        summary["nea_rainfall"]["stopped_before"] == "2025-01-01"
        and session.calls == []
    )


def collection(years: dict[int, str]) -> list:
    meta = FakeResponse(
        200,
        {
            "code": 0,
            "data": {"collectionMetadata": {"childDatasets": list(years.values())}},
        },
    )
    names = {
        did: f"Historical Rainfall across Singapore ({y})" for y, did in years.items()
    }

    def dataset(url, params):
        did = re.search(r"datasets/(d_\w+)/metadata", url).group(1)
        return FakeResponse(
            200, {"code": 0, "data": {"datasetId": did, "name": names[did]}}
        )

    return [("collections/2279/metadata", meta), ("/metadata", dataset)]


def test_api_start_is_the_day_after_the_last_csv_year():
    http, *_ = fake_http(collection({2016: "d_a", 2023: "d_b", 2024: "d_c"}))
    assert api_start(http, RAINFALL) == date(2025, 1, 1)
    assert api_start(http, S.API_SOURCES_BY_NAME["pub_flood_alerts"]) == date(
        2025, 9, 1
    )


def test_hist_csv_lands_published_years_and_reports_the_rest(tmp_path):
    csv = b"date,timestamp,station_id,reading_value\n2024-01-01,2024-01-01T00:00:00+08:00,S1,0\n"
    ready = FakeResponse(
        201,
        {
            "code": 0,
            "data": {"status": "DOWNLOAD_SUCCESS", "url": "https://s3.example/d_c.csv"},
        },
    )
    routes = collection({2024: "d_c"}) + [
        ("initiate-download", FakeResponse(201, {"code": 0, "data": {}})),
        ("poll-download", ready),
        ("s3.example", FakeResponse(200, raw=csv)),
    ]
    http, *_ = fake_http(routes)
    sink = LocalSink(tmp_path / "landing")
    summary = hist_csv(
        http,
        sink,
        sources=[RAINFALL],
        years=[2024, 2025, 2026],
        gzip_files=True,
    )
    assert summary["nea_rainfall_hist"] == {
        "landed": [2024],
        "skipped": [],
        "not_published": [2025, 2026],
    }
    landed = (
        tmp_path
        / "landing"
        / S.hist_csv_path("nea_rainfall_hist", 2024, "d_c", gz=True)
    )
    assert gzip.decompress(landed.read_bytes()) == csv

    again = hist_csv(
        http,
        sink,
        sources=[RAINFALL],
        years=[2024],
        gzip_files=True,
    )
    assert again["nea_rainfall_hist"]["skipped"] == [2024]


def test_telegram_archive_pages_back_to_the_first_post(tmp_path):
    html = (FIXTURES / "telegram_page.html").read_text()
    starts = [
        m.start() for m in re.finditer(r'<div class="tgme_widget_message_wrap', html)
    ]
    end = html.index("</section>")
    wraps = [
        html[s:e] for s, e in zip(starts, starts[1:] + [end], strict=True)
    ]  # posts 1, 4, 1630, 1631

    def page(url, params):
        chosen = wraps[2:] if "before=" not in url else wraps[:2]
        return FakeResponse(200, text_body="<section>" + "".join(chosen) + "</section>")

    http, session, _ = fake_http([("t.me/s/pubfloodalerts", page)])
    summary = telegram_archive(http, LocalSink(tmp_path), runner="test", polite_delay=0)
    assert summary["posts"] == 4 and summary["pages"] == 2
    assert session.calls[1]["url"].endswith("?before=1630")
    assert [p["message_id"] for p in gz_lines(tmp_path / summary["path"])] == [
        1,
        4,
        1630,
        1631,
    ]
