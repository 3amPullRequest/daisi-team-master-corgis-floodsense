"""Static layers land whole, except a line too long for one bronze VARIANT, which lands in parts that keep every
segment."""

import gzip
import json

from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest import static
from daisi_team_master_corgis_floodsense.ingest.client import Http
from daisi_team_master_corgis_floodsense.ingest.sinks import LocalSink
from tests.fakes import FakeClock, FakeResponse, FakeSession


def vertices(n: int, start: int = 0) -> list[list[float]]:
    """n vertices of a line along the equator's latitude of Singapore, each with SLA's z of 0."""
    return [[103.8 + (start + i) * 1e-5, 1.3, 0] for i in range(n)]


def feature(kind: str, coordinates, **properties) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": kind, "coordinates": coordinates},
        "properties": {"OBJECTID": 7, "FOLDERPATH": "Layers/Contour_250K", **properties},
    }


def segments(line: list) -> list[tuple]:
    return [(tuple(a), tuple(b)) for a, b in zip(line, line[1:])]


def test_short_lines_polygons_and_features_without_a_geometry_land_whole():
    short = feature("LineString", vertices(5), NAME="20")
    ring = [*vertices(19), vertices(1)[0]]
    polygon = feature("Polygon", [ring], FOLDERPATH="Layers/Coastal_Outlines")
    blank = {"type": "Feature", "geometry": None, "properties": {"NAME": "MY WATERWAY@PUNGGOL"}}
    for whole in (short, polygon, blank):
        assert static.feature_parts(whole, max_vertices=5) == [whole]


def test_a_long_line_lands_in_parts_that_keep_every_segment_once():
    line = vertices(23)
    contour = feature("LineString", line, NAME="40")
    parts = static.feature_parts(contour, max_vertices=5)
    # 22 segments, 4 to a part, each part a LineString of the contour's properties
    assert len(parts) == 6
    assert all(p["geometry"]["type"] == "LineString" for p in parts)
    assert all(len(p["geometry"]["coordinates"]) <= 5 for p in parts)
    assert all(p["properties"] == contour["properties"] for p in parts)
    # each part starts at the vertex the part before ends at
    for before, after in zip(parts, parts[1:]):
        assert before["geometry"]["coordinates"][-1] == after["geometry"]["coordinates"][0]
    assert [s for p in parts for s in segments(p["geometry"]["coordinates"])] == segments(line)


def test_a_long_multiline_lands_member_by_member():
    members = [vertices(7), vertices(3, start=100)]
    parts = static.feature_parts(feature("MultiLineString", members), max_vertices=5)
    assert all(p["geometry"]["type"] == "LineString" for p in parts)
    assert all(len(p["geometry"]["coordinates"]) <= 5 for p in parts)
    landed = [s for p in parts for s in segments(p["geometry"]["coordinates"])]
    assert landed == segments(members[0]) + segments(members[1])


def test_fetch_static_lands_a_line_per_part_with_its_number(tmp_path, monkeypatch):
    monkeypatch.setattr(static, "MAX_PART_VERTICES", 5)
    collection = {
        "type": "FeatureCollection",
        "features": [feature("LineString", vertices(9), NAME="20"), feature("LineString", vertices(3), NAME="0")],
    }
    session = FakeSession(
        [
            ("/metadata", FakeResponse(payload={"data": {"name": "National Map Line", "lastUpdatedAt": "2026-10-25T10:09:00+08:00"}})),
            ("/poll-download", FakeResponse(payload={"data": {"status": "DOWNLOAD_SUCCESS", "url": "https://s3.example/map.geojson"}})),
            ("s3.example", FakeResponse(raw=json.dumps(collection).encode())),
        ]
    )  # fmt: skip
    clock = FakeClock()
    http = Http(None, session=session, clock=clock, sleep=clock.sleep)
    layer = S.StaticLayer("static_national_map_line", "d_line", "geojson")
    out = static.fetch_static(http, LocalSink(tmp_path), runner="test", layers=[layer])
    assert out == [
        {
            "layer": layer.name,
            "version": "2026-10-25",
            "rows": 2,
            "path": "static_national_map_line/version=2026-10-25/d_line.jsonl.gz",
        }
    ]
    lines = [json.loads(x) for x in gzip.decompress((tmp_path / out[0]["path"]).read_bytes()).splitlines()]
    # the first contour's 8 segments in two parts, the second whole
    assert [(x["feature_index"], x["part"], x["parts"]) for x in lines] == [(0, 0, 2), (0, 1, 2), (1, 0, 1)]
    assert [len(x["feature"]["geometry"]["coordinates"]) for x in lines] == [5, 5, 3]
    assert all(x["dataset_id"] == "d_line" and x["version"] == "2026-10-25" for x in lines)
    # the original file is kept beside it, for audit, under a name Auto Loader doesn't read
    original = tmp_path / "static_national_map_line/version=2026-10-25/d_line.geojson.gz"
    assert json.loads(gzip.decompress(original.read_bytes())) == collection
