"""Static context layers from data.gov.sg (land use, PUB water-level sensors, flood-prone area, SLA benchmarks, the
national map's polygons and lines). Each layer lands under `<name>/version=<lastUpdatedAt date>/`: the original file
(kept for audit, ignored by Auto Loader) and a readable copy, one GeoJSON feature or spreadsheet row per JSONL line,
written last. A layer whose readable copy exists for the current version is skipped, so re-runs only fetch layers that
data.gov.sg updated.

For each layer, `fetch_static` asks data.gov.sg for the dataset's metadata and takes its `lastUpdatedAt` date as the
version. If the readable copy of that version is already there, it moves on. Otherwise it downloads the file (the
presigned-URL handshake in client.py) and lands it twice. The original bytes are kept for audit under a name Auto
Loader doesn't read (.geojson.gz, .xlsx). The readable copy is what bronze streams: for GeoJSON, one JSONL line per
feature (`feature_index`, `part`, `parts`, `feature`); for a spreadsheet, one line per row (`sheet`, `row_index`,
`row`, built by `xlsx_rows` from the first non-empty row of each sheet as the header, every value as text); for CSV,
the file itself. Each line also carries the dataset id, name and version. The readable copy is written last, so its
presence is what marks a version done. A newer version lands beside the old ones, and silver reads the newest.

Bronze holds each line as one VARIANT, of at most 16 MiB. The national map's contours have a vertex every 30 cm or so,
which puts up to about 900,000 vertices in one line (40 MB of JSON). So a line feature of more than MAX_PART_VERTICES
vertices lands as several lines, its consecutive parts (`feature_parts`): each a LineString of the same properties that
starts at the vertex the part before it ends at, so no vertex or segment is lost. Every other feature lands whole, as
part 0 of 1.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from collections.abc import Sequence
from datetime import date, datetime

from . import sources as S
from .cli import add_common_args, setup
from .client import Http, dataset_file_url, dataset_metadata
from .sinks import Sink

log = logging.getLogger(__name__)

# Vertices a landed line may hold: about 2 MB of JSON, well inside bronze's 16 MiB VARIANT.
MAX_PART_VERTICES = 50_000


def readable_ext(kind: str) -> str:
    return {"geojson": "jsonl.gz", "xlsx": "jsonl", "csv": "csv"}[kind]


def feature_parts(feature: dict, max_vertices: int = MAX_PART_VERTICES) -> list[dict]:
    """The feature as it lands: whole, or, for a LineString or MultiLineString of more than `max_vertices` vertices,
    its consecutive parts as LineStrings (each member line in turn, cut every `max_vertices - 1` segments), which keep
    the feature's properties. A part starts at the vertex the part before it ends at, so every segment is in exactly
    one part. Polygons and points always land whole."""
    geometry = feature.get("geometry") or {}
    kind = geometry.get("type")
    if kind == "LineString":
        lines = [geometry.get("coordinates") or []]
    elif kind == "MultiLineString":
        lines = geometry.get("coordinates") or []
    else:
        return [feature]
    if sum(len(line) for line in lines) <= max_vertices:
        return [feature]
    step = max_vertices - 1
    return [
        {**feature, "geometry": {"type": "LineString", "coordinates": line[i : i + max_vertices]}}
        for line in lines
        for i in range(0, max(len(line) - 1, 1), step)
    ]


def fetch_static(
    http: Http,
    sink: Sink,
    *,
    runner: str,
    layers: Sequence[S.StaticLayer] = S.STATIC_LAYERS,
    force: bool = False,
) -> list[dict]:
    out = []
    for layer in layers:
        meta = dataset_metadata(http, layer.dataset_id)
        version = (meta.get("lastUpdatedAt") or "")[:10] or "unknown"
        base = f"{S.static_folder(layer.name, version)}/{layer.dataset_id}"
        readable = f"{base}.{readable_ext(layer.kind)}"
        if sink.exists(readable) and not force:
            out.append({"layer": layer.name, "version": version, "skipped": True})
            continue
        raw = http.get(
            dataset_file_url(http, layer.dataset_id, initiate=layer.kind == "csv")
        ).content
        fetched = S.utcnow()
        common = {
            "dataset_id": layer.dataset_id,
            "dataset_name": meta.get("name"),
            "version": version,
        }
        if layer.kind == "csv":
            sink.write_bytes(readable, raw)
            count = max(0, raw.count(b"\n") - 1)
        elif layer.kind == "geojson":
            sink.write_bytes(f"{base}.geojson.gz", S.gzipped(raw))
            features = json.loads(raw).get("features") or []
            lines = (
                S.record(
                    layer.name,
                    fetched_at=fetched,
                    runner=runner,
                    feature_index=i,
                    part=k,
                    parts=len(parts),
                    feature=part,
                    **common,
                )
                for i, f in enumerate(features)
                for parts in (feature_parts(f, MAX_PART_VERTICES),)
                for k, part in enumerate(parts)
            )
            sink.write_bytes(readable, S.gzipped(S.jsonl(lines)))
            count = len(features)
        else:
            sink.write_bytes(f"{base}.xlsx", raw)
            rows = xlsx_rows(raw)
            lines = (
                S.record(
                    layer.name,
                    fetched_at=fetched,
                    runner=runner,
                    sheet=sheet,
                    row_index=i,
                    row=row,
                    **common,
                )
                for i, (sheet, row) in enumerate(rows)
            )
            sink.write_bytes(readable, S.jsonl(lines))
            count = len(rows)
        log.info(
            "%s %s: %d rows from %.1f MB", layer.name, version, count, len(raw) / 1e6
        )
        out.append(
            {"layer": layer.name, "version": version, "rows": count, "path": readable}
        )
    return out


def xlsx_rows(raw: bytes) -> list[tuple[str, dict]]:
    """(sheet, {header: text value}) for every non-empty row under each sheet's first non-empty row."""
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    rows = []
    for sheet in workbook.worksheets:
        header = None
        for values in sheet.iter_rows(values_only=True):
            if all(v is None for v in values):
                continue
            if header is None:
                header = [
                    str(v).strip() if v is not None else f"column_{i}"
                    for i, v in enumerate(values)
                ]
                continue
            rows.append(
                (
                    sheet.title,
                    {h: _text(v) for h, v in zip(header, values, strict=False)},
                )
            )
    return rows


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return repr(value) if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser)
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=[layer.name for layer in S.STATIC_LAYERS],
        help="default: all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="land again even if this version has landed",
    )
    args = parser.parse_args(argv)
    http, sink = setup(args, share=0.6)
    layers = [
        layer
        for layer in S.STATIC_LAYERS
        if not args.layers or layer.name in args.layers
    ]
    print(
        json.dumps(
            fetch_static(
                http, sink, runner=args.runner, layers=layers, force=args.force
            ),
            indent=1,
        )
    )
    return 0
