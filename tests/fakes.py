"""Test doubles: a scripted requests session and a manual clock, so ingestion tests never touch the network."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from daisi_team_master_corgis_floodsense.ingest.client import Http


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: object = None
    text_body: str | None = None
    raw: bytes | None = None
    headers: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.text_body is not None:
            return self.text_body
        return json.dumps(self.payload) if self.payload is not None else ""

    @property
    def content(self) -> bytes:
        return self.raw if self.raw is not None else self.text.encode()

    def json(self):
        return self.payload if self.payload is not None else json.loads(self.text)

    def iter_content(self, chunk_size: int = 1024):
        data = self.content
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Answers GETs from `routes`: (url substring, response or callable(url, params) -> response), first match
    wins; a list of responses is consumed in order. Every call is recorded in `calls`."""

    def __init__(self, routes: list[tuple[str, object]]):
        self.routes = routes
        self.headers: dict = {}
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None, stream=False):
        self.calls.append(
            {"url": url, "params": dict(params or {}), "headers": dict(headers or {})}
        )
        for needle, answer in self.routes:
            if needle in url:
                if isinstance(answer, list):
                    return answer.pop(0)
                if callable(answer):
                    return answer(url, params or {})
                return answer
        raise AssertionError(f"unexpected GET {url} {params}")


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def fake_http(
    routes, *, api_key: str | None = None, share: float = 1.0
) -> tuple[Http, FakeSession, FakeClock]:
    clock = FakeClock()
    session = FakeSession(routes)
    http = Http(api_key, session=session, share=share, clock=clock, sleep=clock.sleep)
    return http, session, clock


def api_page(
    records_key: str, n: int, token: str | None = None, **extra
) -> FakeResponse:
    data = {
        records_key: [{"timestamp": f"t{i}"} for i in range(n)],
        "paginationToken": token,
        **extra,
    }
    return FakeResponse(200, {"code": 0, "data": data, "errorMsg": ""})


def no_data() -> FakeResponse:
    return FakeResponse(
        404,
        {
            "code": 17,
            "name": "REAL_TIME_API_DATA_NOT_FOUND",
            "data": None,
            "errorMsg": "Data not found",
        },
    )
