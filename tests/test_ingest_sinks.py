import gzip
import io
from types import SimpleNamespace

import pytest
from databricks.sdk.errors import NotFound

from daisi_team_master_corgis_floodsense.ingest.sinks import (
    FilesApiSink,
    LocalSink,
    make_sink,
    read_json,
    write_json,
)


def test_local_sink_round_trip(tmp_path):
    sink = LocalSink(tmp_path)
    sink.write_bytes("a/date=2026-10-10/x.jsonl", b"{}\n")
    assert sink.exists("a/date=2026-10-10/x.jsonl")
    assert sink.read_bytes("a/date=2026-10-10/x.jsonl") == b"{}\n"
    assert sink.read_bytes("missing") is None
    assert sink.list_names("a") == ["date=2026-10-10"]
    assert sink.list_names("missing") == []
    # nothing left behind under a temp name
    assert [p.name for p in (tmp_path / "a/date=2026-10-10").iterdir()] == ["x.jsonl"]


def test_local_sink_streams_plain_or_gzipped(tmp_path):
    sink = LocalSink(tmp_path)
    assert sink.write_stream("t/year=2024/d.csv", [b"a,b\n", b"1,2\n"]) == 8
    assert sink.read_bytes("t/year=2024/d.csv") == b"a,b\n1,2\n"
    sink.write_stream("t/year=2024/d.csv.gz", [b"a,b\n", b"1,2\n"], compress=True)
    assert gzip.decompress(sink.read_bytes("t/year=2024/d.csv.gz")) == b"a,b\n1,2\n"
    assert sorted(p.name for p in (tmp_path / "t/year=2024").iterdir()) == [
        "d.csv",
        "d.csv.gz",
    ]


def test_local_sink_leaves_nothing_when_a_stream_fails(tmp_path):
    def broken():
        yield b"partial"
        raise ConnectionError("dropped")

    sink = LocalSink(tmp_path)
    with pytest.raises(ConnectionError):
        sink.write_stream("t/x.csv", broken())
    assert list((tmp_path / "t").iterdir()) == []


def test_state_helpers(tmp_path):
    sink = LocalSink(tmp_path)
    assert read_json(sink, "_state/x.json") == {}
    write_json(sink, "_state/x.json", {"last": 3})
    assert read_json(sink, "_state/x.json") == {"last": 3}


def test_auto_sink_choice(tmp_path):
    assert isinstance(make_sink(str(tmp_path)), LocalSink)
    assert isinstance(make_sink("./not-there-yet"), LocalSink)


class _FakeFiles:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def upload(self, path, contents, overwrite=None):
        self.store[path] = contents.read()

    def get_metadata(self, path):
        if path not in self.store:
            raise NotFound("missing")

    def download(self, path):
        if path not in self.store:
            raise NotFound("missing")
        return SimpleNamespace(contents=io.BytesIO(self.store[path]))

    def list_directory_contents(self, path):
        names = {
            p[len(path) + 1 :].split("/")[0]
            for p in self.store
            if p.startswith(path + "/")
        }
        if not names:
            raise NotFound("missing")
        return [SimpleNamespace(name=n) for n in names]


def test_files_api_sink_maps_to_volume_paths(tmp_path):
    files = _FakeFiles()
    sink = FilesApiSink(
        "/Volumes/c/s/landing/", SimpleNamespace(files=files), tmp_dir=tmp_path
    )
    sink.write_bytes("t/x.jsonl", b"1\n")
    assert files.store == {"/Volumes/c/s/landing/t/x.jsonl": b"1\n"}
    sink.write_stream("t/big.csv.gz", [b"a\n", b"b\n"], compress=True)
    assert (
        gzip.decompress(files.store["/Volumes/c/s/landing/t/big.csv.gz"]) == b"a\nb\n"
    )
    assert sink.exists("t/x.jsonl") and not sink.exists("t/y.jsonl")
    assert sink.read_bytes("t/x.jsonl") == b"1\n" and sink.read_bytes("nope") is None
    assert (
        sink.list_names("t") == ["big.csv.gz", "x.jsonl"] and sink.list_names("u") == []
    )
