import json
from datetime import UTC, datetime
from pathlib import Path

from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest.live import daily_repair, run_tick
from daisi_team_master_corgis_floodsense.ingest.sinks import LocalSink
from tests.fakes import FakeResponse, api_page, fake_http

FIXTURES = Path(__file__).parent / "fixtures"
PNG = b"\x89PNG\r\n\x1a\nfake"


def radar_page(*stamps: str) -> FakeResponse:
    records = [
        {"timestamp": ts, "image": {"url": f"https://s3.example/{ts}.png"}}
        for ts in stamps
    ]
    return FakeResponse(
        200,
        {
            "code": 0,
            "data": {"projection": {}, "records": records, "paginationToken": None},
        },
    )


def routes(radar: FakeResponse) -> list:
    # first match wins, so the more specific "weather..." endpoints come first
    return [
        ("weather-radar-images", radar),
        ("s3.example", FakeResponse(200, raw=PNG)),
        ("weather/flood-alerts", api_page("records", 25)),
        ("real-time/api/weather", api_page("records", 25)),  # lightning
        ("two-hr-forecast", api_page("items", 25, area_metadata=[])),
        ("real-time/api/", api_page("readings", 25, stations=[])),
        (
            "t.me/s/pubfloodalerts",
            FakeResponse(200, text_body=(FIXTURES / "telegram_page.html").read_text()),
        ),
        (
            "nea.gov.sg",
            FakeResponse(200, text_body=(FIXTURES / "nea_tides.html").read_text()),
        ),
    ]


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tick_lands_every_source_and_a_heartbeat(tmp_path):
    http, session, _ = fake_http(
        routes(radar_page("2026-10-10T18:45:00+08:00", "2026-10-10T18:50:00+08:00"))
    )
    sink = LocalSink(tmp_path)
    now = datetime(2026, 10, 10, 10, 50, tzinfo=UTC)
    heartbeat = run_tick(http, sink, runner="test", now=now)

    assert heartbeat["ok"], heartbeat
    for source in S.API_SOURCES:
        (live_file,) = (tmp_path / source.name / "live" / "date=2026-10-10").iterdir()
        (envelope,) = lines(live_file)
        assert (
            envelope["status"] == 200
            and envelope["request"]["params"]["date"] == "2026-10-10"
        )
    lightning = next(c for c in session.calls if c["url"].endswith("/weather"))
    assert lightning["params"] == {"api": "lightning", "date": "2026-10-10"}

    assert sorted(
        p.name for p in (tmp_path / "radar_png/70km/2026/10/10").iterdir()
    ) == [
        "rain_70km_202610101845.png",
        "rain_70km_202610101850.png",
    ]
    (posts,) = (tmp_path / S.TELEGRAM / "live" / "date=2026-10-10").iterdir()
    assert [p["message_id"] for p in lines(posts)] == [1, 4, 1630, 1631]
    (tides,) = (tmp_path / S.TIDES / "daily" / "date=2026-10-10").iterdir()
    assert len(lines(tides)) == 16
    (beat,) = (tmp_path / S.HEARTBEAT / "ticks" / "date=2026-10-10").iterdir()
    assert {r["source"] for r in lines(beat)[0]["results"]} == {
        s.name for s in S.API_SOURCES
    } | {S.TELEGRAM, S.TIDES}


def test_second_tick_lands_only_what_is_new(tmp_path):
    sink = LocalSink(tmp_path)
    http, *_ = fake_http(routes(radar_page("2026-10-10T18:45:00+08:00")))
    run_tick(http, sink, runner="test", now=datetime(2026, 10, 10, 10, 45, tzinfo=UTC))
    http, session, _ = fake_http(
        routes(radar_page("2026-10-10T18:45:00+08:00", "2026-10-10T18:50:00+08:00"))
    )
    heartbeat = run_tick(
        http, sink, runner="test", now=datetime(2026, 10, 10, 10, 50, tzinfo=UTC)
    )

    results = {r["source"]: r for r in heartbeat["results"]}
    assert results[S.RADAR]["frames"] == 1  # only the 18:50 frame
    assert results[S.TELEGRAM]["records"] == 0 and "path" not in results[S.TELEGRAM]
    assert results[S.TIDES]["skipped"]
    assert not any("nea.gov.sg" in c["url"] for c in session.calls)
    assert (
        len(list((tmp_path / S.TELEGRAM / "live" / "date=2026-10-10").iterdir())) == 1
    )


def test_a_failing_source_does_not_stop_the_tick(tmp_path):
    broken = [("two-hr-forecast", FakeResponse(400, {"errorMsg": "bad"}))] + routes(
        radar_page()
    )
    http, *_ = fake_http(broken)
    heartbeat = run_tick(
        http,
        LocalSink(tmp_path),
        runner="test",
        now=datetime(2026, 10, 10, 10, 50, tzinfo=UTC),
    )
    results = {r["source"]: r for r in heartbeat["results"]}
    assert not heartbeat["ok"]
    assert (
        not results["nea_two_hr_forecast"]["ok"]
        and "HttpError" in results["nea_two_hr_forecast"]["error"]
    )
    assert results["nea_rainfall"]["ok"]


def test_daily_repair_runs_once_after_half_past_midnight(tmp_path):
    sink = LocalSink(tmp_path)
    http, session, _ = fake_http(routes(radar_page()))
    before_0030 = datetime(2026, 10, 10, 16, 15, tzinfo=UTC)  # 00:15 SGT
    after_0030 = datetime(2026, 10, 10, 16, 45, tzinfo=UTC)  # 00:45 SGT
    assert daily_repair(http, sink, runner="test", now=before_0030) is None
    summary = daily_repair(http, sink, runner="test", now=after_0030)
    assert set(summary) == {s.name for s in S.API_SOURCES}
    assert (
        tmp_path / S.backfill_path("nea_rainfall", datetime(2026, 10, 10).date())
    ).exists()
    calls = len(session.calls)
    assert (
        daily_repair(
            http, sink, runner="test", now=datetime(2026, 10, 10, 17, 0, tzinfo=UTC)
        )
        is None
    )
    assert len(session.calls) == calls
