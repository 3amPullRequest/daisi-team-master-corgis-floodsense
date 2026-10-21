"""SQL on the app's warehouse (the Statement Execution API, as the app's service principal on Databricks or your
profile locally), with a small time-based cache so a page refresh doesn't start a query."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    Disposition,
    Format,
    StatementParameterListItem,
    StatementState,
)

_client: WorkspaceClient | None = None
_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


def client() -> WorkspaceClient:
    global _client
    with _lock:
        if _client is None:
            _client = (
                WorkspaceClient()
            )  # app: the injected service principal; locally: your profile
        return _client


class QueryError(RuntimeError):
    pass


def query(
    warehouse_id: str,
    sql: str,
    params: dict[str, str] | None = None,
    timeout_s: int = 120,
) -> list[dict]:
    """Rows as dicts, typed from the result manifest. A cold serverless warehouse can take a minute to start, so a
    statement that hasn't finished in 30 s is polled until `timeout_s`."""
    if not warehouse_id:
        raise QueryError("no SQL warehouse configured (DATABRICKS_WAREHOUSE_ID)")
    w = client()
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        parameters=[
            StatementParameterListItem(name=k, value=v)
            for k, v in (params or {}).items()
        ],
        wait_timeout="30s",
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
    )
    deadline = time.monotonic() + timeout_s
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.monotonic() > deadline:
            w.statement_execution.cancel_execution(resp.statement_id)
            raise QueryError("the warehouse took too long to answer")
        time.sleep(1.0)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        detail = resp.status.error.message if resp.status.error else resp.status.state
        raise QueryError(f"query failed: {detail}")
    columns = [
        (c.name, c.type_name.value if c.type_name else "STRING")
        for c in resp.manifest.schema.columns
    ]
    rows = (resp.result.data_array if resp.result else None) or []
    return [
        {name: _typed(value, kind) for (name, kind), value in zip(columns, row)}
        for row in rows
    ]


def _typed(value: str | None, kind: str):
    if value is None:
        return None
    if kind in ("INT", "LONG", "SHORT", "BYTE"):
        return int(value)
    if kind in ("DOUBLE", "FLOAT", "DECIMAL"):
        return float(value)
    if kind == "BOOLEAN":
        return value.lower() == "true"
    if kind == "TIMESTAMP":
        # the API answers in UTC ('2026-10-21T08:25:00.000Z'); the browser shows SGT
        return datetime.fromisoformat(value).isoformat()
    return value


def fresh(key: str, ttl_s: float) -> Any:
    """The cached value for `key` if it is younger than `ttl_s` seconds, else None."""
    hit = _cache.get(key)
    return hit[1] if hit and time.monotonic() - hit[0] < ttl_s else None


def put(key: str, value: Any) -> Any:
    _cache[key] = (time.monotonic(), value)
    return value


def cached(key: str, ttl_s: float, load: Callable[[], Any]) -> Any:
    """The value for `key`, reloaded at most every `ttl_s` seconds. A failed reload raises; the caller decides
    whether a stale value is better than an error."""
    hit = fresh(key, ttl_s)
    return hit if hit is not None else put(key, load())


def stale(key: str) -> Any:
    """The last value cached for `key`, however old (to show while a reload fails)."""
    hit = _cache.get(key)
    return hit[1] if hit else None
