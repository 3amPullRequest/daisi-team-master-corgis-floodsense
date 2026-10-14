"""Where landed files go: a folder (the /Volumes mount on Databricks, or a local dir), or the UC volume through the
Files API when running outside Databricks."""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO, Protocol

log = logging.getLogger(__name__)


class Sink(Protocol):
    def write_bytes(self, rel_path: str, data: bytes) -> None: ...
    def write_stream(
        self, rel_path: str, chunks: Iterable[bytes], *, compress: bool = False
    ) -> int: ...
    def exists(self, rel_path: str) -> bool: ...
    def read_bytes(self, rel_path: str) -> bytes | None: ...
    def list_names(self, rel_dir: str) -> list[str]: ...


class LocalSink:
    """Writes under `root`. Each file is written to a hidden temp name first (Auto Loader skips dot-files), then
    renamed into place, so readers never see a partial file."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __str__(self) -> str:
        return f"folder {self.root}"

    def _dest(self, rel_path: str) -> tuple[Path, Path]:
        dest = self.root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest, dest.with_name(f".{dest.name}.{os.getpid()}.tmp")

    def write_bytes(self, rel_path: str, data: bytes) -> None:
        dest, tmp = self._dest(rel_path)
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    def write_stream(
        self, rel_path: str, chunks: Iterable[bytes], *, compress: bool = False
    ) -> int:
        """Stream straight into the destination folder (on Databricks, the volume), gzipping if asked."""
        dest, tmp = self._dest(rel_path)
        try:
            size = _write_chunks(tmp, chunks, compress)
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return size

    def exists(self, rel_path: str) -> bool:
        return (self.root / rel_path).exists()

    def read_bytes(self, rel_path: str) -> bytes | None:
        try:
            return (self.root / rel_path).read_bytes()
        except FileNotFoundError:
            return None

    def list_names(self, rel_dir: str) -> list[str]:
        try:
            return sorted(
                p.name
                for p in (self.root / rel_dir).iterdir()
                if not p.name.startswith(".")
            )
        except FileNotFoundError:
            return []


class FilesApiSink:
    """Writes into a UC volume (`/Volumes/<catalog>/<schema>/<volume>/...`) with the Databricks Files API."""

    def __init__(self, root: str, client, *, tmp_dir: Path | None = None):
        from databricks.sdk.errors import NotFound

        self.root = root.rstrip("/")
        self.client = client
        # Large files are staged on disk before upload; /tmp is often RAM-backed (tmpfs), so default elsewhere.
        self.tmp_dir = tmp_dir or Path(
            os.environ.get(
                "FLOODSENSE_TMP", Path.home() / ".cache" / "floodsense-ingest"
            )
        )
        self._not_found = NotFound

    def __str__(self) -> str:
        return f"Files API {self.root} ({self.client.config.host})"

    def _path(self, rel_path: str) -> str:
        return f"{self.root}/{rel_path}"

    def write_bytes(self, rel_path: str, data: bytes) -> None:
        self.client.files.upload(self._path(rel_path), io.BytesIO(data), overwrite=True)

    def write_stream(
        self, rel_path: str, chunks: Iterable[bytes], *, compress: bool = False
    ) -> int:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp:
            local = Path(tmp) / Path(rel_path).name
            size = _write_chunks(local, chunks, compress)
            with open(local, "rb") as f:
                self.client.files.upload(self._path(rel_path), f, overwrite=True)
        return size

    def exists(self, rel_path: str) -> bool:
        try:
            self.client.files.get_metadata(self._path(rel_path))
            return True
        except self._not_found:
            return False

    def read_bytes(self, rel_path: str) -> bytes | None:
        try:
            return self.client.files.download(self._path(rel_path)).contents.read()
        except self._not_found:
            return None

    def list_names(self, rel_dir: str) -> list[str]:
        try:
            return sorted(
                e.name
                for e in self.client.files.list_directory_contents(self._path(rel_dir))
            )
        except self._not_found:
            return []


def make_sink(landing: str, kind: str = "auto", profile: str | None = None) -> Sink:
    """`auto` writes straight to the folder when it exists or is local (on Databricks /Volumes is mounted), and uses
    the Files API for a /Volumes path that isn't mounted here."""
    if kind == "auto":
        kind = (
            "files-api"
            if landing.startswith("/Volumes/") and not Path(landing).exists()
            else "local"
        )
    if kind == "local":
        return LocalSink(landing)
    from databricks.sdk import WorkspaceClient

    return FilesApiSink(
        landing, WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    )


def _write_chunks(path: Path, chunks: Iterable[bytes], compress: bool) -> int:
    out: BinaryIO
    with (
        gzip.open(path, "wb", compresslevel=6) if compress else open(path, "wb")
    ) as out:
        for chunk in chunks:
            out.write(chunk)
    return path.stat().st_size


def read_json(sink: Sink, rel_path: str) -> dict:
    data = sink.read_bytes(rel_path)
    return json.loads(data) if data else {}


def write_json(sink: Sink, rel_path: str, obj: dict) -> None:
    sink.write_bytes(rel_path, json.dumps(obj, indent=1).encode())
