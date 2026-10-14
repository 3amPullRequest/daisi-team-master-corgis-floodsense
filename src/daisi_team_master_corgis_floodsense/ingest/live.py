"""One live tick: the newest page of every API source, new Telegram posts, new radar frames, and (once a day) NEA's
tide table, then a heartbeat line summarising the tick. `main` can repeat ticks on an interval."""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from datetime import time as clock_time

from . import sources as S
from .cli import add_common_args, parse_duration, setup
from .client import ApiResponse, Http, realtime
from .history import api_days, archive_radar_frames
from .scrapers import parse_nea_tides, parse_telegram_page
from .sinks import Sink, read_json, write_json

log = logging.getLogger(__name__)

LIVE_NAMES = [s.name for s in S.API_SOURCES] + [S.TELEGRAM, S.TIDES]
REPAIR_AFTER = clock_time(0, 30)  # SGT; yesterday's last readings have settled by then


def newest_page(http: Http, source: S.ApiSource, now: datetime) -> ApiResponse:
    """Page 1 of today's readings (newest 25 records), so a late or skipped tick fills its own gap. Just after
    midnight SGT the new day can still be empty, so fall back to the plain latest reading."""
    resp = realtime(
        http, source.endpoint, {**source.query, "date": S.sgt_today(now).isoformat()}
    )
    return (
        resp
        if resp.status == 200
        else realtime(http, source.endpoint, source.query or None)
    )


def run_tick(
    http: Http,
    sink: Sink,
    *,
    runner: str,
    names: set[str] | None = None,
    now: datetime | None = None,
) -> dict:
    started = now or S.utcnow()
    results = []
    for source in S.API_SOURCES:
        if names is None or source.name in names:
            results.append(
                _guard(
                    source.name,
                    lambda s=source: _land_api(http, sink, s, started, runner),
                )
            )
    if names is None or S.TELEGRAM in names:
        results.append(
            _guard(S.TELEGRAM, lambda: _land_telegram(http, sink, started, runner))
        )
    if names is None or S.TIDES in names:
        results.append(
            _guard(S.TIDES, lambda: _land_tides(http, sink, started, runner))
        )
    heartbeat = S.record(
        S.HEARTBEAT,
        fetched_at=started,
        runner=runner,
        finished_at=S.iso(S.utcnow()),
        ok=all(r["ok"] for r in results),
        results=results,
    )
    sink.write_bytes(S.run_path(S.HEARTBEAT, "ticks", started), S.jsonl([heartbeat]))
    failed = [r["source"] for r in results if not r["ok"]]
    log.info(
        "tick %s: %d sources ok%s",
        S.stamp(started),
        len(results) - len(failed),
        f", FAILED: {failed}" if failed else "",
    )
    return heartbeat


def _guard(name: str, land: Callable[[], dict]) -> dict:
    t0 = time.monotonic()
    try:
        result = {"ok": True, **land()}
    except Exception as exc:
        log.exception("%s failed", name)
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    return {
        "source": name,
        **result,
        "latency_ms": round((time.monotonic() - t0) * 1000),
    }


def _land_api(
    http: Http, sink: Sink, source: S.ApiSource, started: datetime, runner: str
) -> dict:
    resp = newest_page(http, source, started)
    path = S.run_path(source.name, "live", started)
    sink.write_bytes(
        path,
        S.jsonl(
            [
                S.api_envelope(
                    source.name, resp, page=1, fetched_at=started, runner=runner
                )
            ]
        ),
    )
    result = {
        "ok": resp.status == 200,
        "status": resp.status,
        "records": S.record_count(resp.body),
        "path": path,
    }
    if source.name == S.RADAR and resp.status == 200:
        cursor = S.state_path("radar_70km")
        saved, newest = archive_radar_frames(
            http,
            sink,
            resp.body,
            after=read_json(sink, cursor).get("last_timestamp", ""),
        )
        if saved:
            write_json(sink, cursor, {"last_timestamp": newest})
        result["frames"] = saved
    return result


def _land_telegram(http: Http, sink: Sink, started: datetime, runner: str) -> dict:
    """Posts newer than the last one landed. Capturing them live also keeps posts PUB later deletes."""
    posts = parse_telegram_page(http.get(S.TELEGRAM_URL).text)
    cursor = S.state_path("telegram")
    last_id = read_json(sink, cursor).get("last_message_id", 0)
    new = [p for p in posts if p["message_id"] > last_id]
    result = {"records": len(new), "page_posts": len(posts)}
    if new:
        path = S.run_path(S.TELEGRAM, "live", started)
        lines = [
            S.record(
                S.TELEGRAM,
                fetched_at=started,
                runner=runner,
                page_url=S.TELEGRAM_URL,
                **p,
            )
            for p in new
        ]
        sink.write_bytes(path, S.jsonl(lines))
        write_json(sink, cursor, {"last_message_id": max(p["message_id"] for p in new)})
        result["path"] = path
    return result


def _land_tides(http: Http, sink: Sink, started: datetime, runner: str) -> dict:
    """NEA's tide table (this month and next), at most once per SGT day."""
    today = S.sgt_today(started).isoformat()
    cursor = S.state_path("nea_tides")
    if read_json(sink, cursor).get("last_day") == today:
        return {"skipped": True}
    rows = parse_nea_tides(http.get(S.NEA_TIDES_URL).text)
    if not rows:
        raise RuntimeError("no tide rows found on the NEA page")
    path = S.run_path(S.TIDES, "daily", started)
    lines = [
        S.record(
            S.TIDES, fetched_at=started, runner=runner, page_url=S.NEA_TIDES_URL, **row
        )
        for row in rows
    ]
    sink.write_bytes(path, S.jsonl(lines))
    write_json(sink, cursor, {"last_day": today})
    return {"records": len(rows), "path": path}


def daily_repair(
    http: Http, sink: Sink, *, runner: str, now: datetime | None = None
) -> dict | None:
    """After 00:30 SGT, re-pull yesterday in full for every API source (once per day)."""
    sgt_now = (now or S.utcnow()).astimezone(S.SGT)
    yesterday = sgt_now.date() - timedelta(days=1)
    cursor = S.state_path("daily_repair")
    if (
        sgt_now.time() < REPAIR_AFTER
        or read_json(sink, cursor).get("last_day") == yesterday.isoformat()
    ):
        return None
    log.info("daily repair: pulling all API pages for %s", yesterday)
    summary = api_days(
        http, sink, sources=S.API_SOURCES, runner=runner, start=yesterday, end=yesterday
    )
    write_json(sink, cursor, {"last_day": yesterday.isoformat()})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=LIVE_NAMES,
        help="only these sources (default: all)",
    )
    parser.add_argument(
        "--every",
        type=parse_duration,
        help="keep running, one tick per interval, e.g. 5m",
    )
    parser.add_argument(
        "--offset",
        type=parse_duration,
        default=60.0,
        help="ticks run this long after each interval boundary (default 60s)",
    )
    parser.add_argument(
        "--daily-repair",
        action="store_true",
        help="after 00:30 SGT, re-pull yesterday's API pages in full",
    )
    parser.add_argument(
        "--trigger-job",
        type=int,
        metavar="JOB_ID",
        help="Databricks job to run-now after each tick",
    )
    args = parser.parse_args(argv)
    http, sink = setup(args, share=0.8)
    names = set(args.sources) if args.sources else None

    def tick() -> bool:
        heartbeat = run_tick(http, sink, runner=args.runner, names=names)
        if args.daily_repair:
            try:
                daily_repair(http, sink, runner=args.runner)
            except Exception:
                log.exception(
                    "daily repair failed; it will be retried after the next tick"
                )
        if args.trigger_job:
            _trigger(sink, args.profile, args.trigger_job)
        return heartbeat["ok"]

    if not args.every:
        return 0 if tick() else 1
    log.info(
        "running a tick every %.0fs (+%.0fs); Ctrl-C to stop", args.every, args.offset
    )
    try:
        while True:
            try:
                tick()
            except Exception:
                log.exception("tick failed")
            boundary = math.floor((time.time() - args.offset) / args.every) + 1
            time.sleep(max(1.0, boundary * args.every + args.offset - time.time()))
    except KeyboardInterrupt:
        return 0


def _trigger(sink: Sink, profile: str | None, job_id: int) -> None:
    try:
        client = getattr(sink, "client", None)
        if client is None:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        run = client.jobs.run_now(job_id=job_id)
        log.info("triggered job %d (run %s)", job_id, run.run_id)
    except Exception:
        log.exception("could not trigger job %d", job_id)
