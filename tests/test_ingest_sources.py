import gzip
import json
from datetime import UTC, date, datetime

from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest.client import ApiResponse


def test_paths_partition_by_singapore_day():
    # 01:30 on the 11th in Singapore
    late_utc = datetime(2026, 10, 10, 17, 30, tzinfo=UTC)
    assert S.run_path("nea_rainfall", "live", late_utc) == (
        "nea_rainfall/live/date=2026-10-11/nea_rainfall_20261010T173000Z.jsonl"
    )
    assert S.backfill_path("nea_rainfall", date(2025, 1, 1)) == (
        "nea_rainfall/backfill/date=2025-01-01/nea_rainfall_2025-01-01.jsonl.gz"
    )
    assert (
        S.hist_csv_path("nea_rainfall_hist", 2024, "d_x", gz=True)
        == "nea_rainfall_hist/year=2024/d_x.csv.gz"
    )


def test_radar_png_path_uses_sgt_like_nea():
    assert (
        S.radar_png_path("2026-10-10T18:50:00+08:00")
        == "radar_png/70km/2026/10/10/rain_70km_202610101850.png"
    )


def test_envelope_keeps_the_body_unchanged():
    body = {"code": 0, "data": {"readings": [{"timestamp": "t", "data": []}]}}
    resp = ApiResponse("https://api/rainfall", {"date": "2026-10-10"}, 200, body)
    env = S.api_envelope(
        "nea_rainfall",
        resp,
        page=2,
        fetched_at=datetime(2026, 10, 10, tzinfo=UTC),
        runner="test",
    )
    assert env["body"] is body
    assert env["request"] == {
        "url": "https://api/rainfall",
        "params": {"date": "2026-10-10"},
        "page": 2,
    }
    assert env["fetched_at"] == "2026-10-10T00:00:00.000Z"


def test_jsonl_round_trip_through_gzip():
    rows = [{"a": 1, "text": "Bukit Timah — flood"}, {"a": 2}]
    data = S.gzipped(S.jsonl(rows))
    assert [
        json.loads(line) for line in gzip.decompress(data).decode().splitlines()
    ] == rows


def test_record_count_finds_the_list():
    assert S.record_count({"data": {"readings": [1, 2]}}) == 2
    assert S.record_count({"data": {"items": [1]}}) == 1
    assert S.record_count({"data": {"records": []}}) == 0
    assert S.record_count({"code": 17, "data": None}) == 0


def test_every_api_source_is_backfillable():
    for source in S.API_SOURCES:
        assert source.csv_collection or source.api_start, source.name
    assert S.API_SOURCES[0].name == "pub_flood_alerts"  # labels first
