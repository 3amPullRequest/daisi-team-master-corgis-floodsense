"""Backfill from 2021 up to yesterday: yearly CSVs where data.gov.sg publishes them, v2 API day pages for every
later day, the Telegram archive (plus the posts PUB deleted, from web archive captures) and the UHSLC tide gauge
record. Each step skips what has already landed.

Resuming works because every unit of work lands as one file, written last, and a re-run skips any unit whose file
exists. So a run can be stopped anywhere, by --max-runtime (a clean stop at a deadline, for a time-limited Databricks
task or an overnight laptop run) or by a crash, and the next run carries on without redoing or duplicating anything.
The subcommands of `main`:

- csv (`hist_csv`): for each source with a CSV collection, ask data.gov.sg which years are published and land each
  year's file under `<name>_hist/year=<y>/`, streamed through the sink and gzipped when uploading. A year that isn't
  published yet is reported and left to api-days.
- api-days (`api_days`): for each API source in priority order, from the day after its last CSV year (or its own start
  date) to yesterday SGT, land every page of each day as one gzipped file under `<name>/backfill/`. A day with no data
  still gets a file, holding the 404 or the empty page, so it isn't asked for again. Radar days also save each frame's
  PNG on the spot (`archive_radar_frames`, shared with live.py), because its presigned URL expires about 25 minutes
  after the API answered. live.py's daily repair calls `api_days` for yesterday.
- telegram (`telegram_archive`): page back through t.me/s/pubfloodalerts with ?before=<oldest id seen> to the oldest
  post the channel still shows, and land one file with all of them. Its summary counts posts per year, a quick check
  for gaps.
- telegram-wayback and onemap hand over to wayback.py and onemap.py; uhslc downloads the tide gauge CSV once and keeps
  its row count in a cursor.

A backfill uses 60% of the key's rate limit (live ticks use 80%), and because the sources are listed by priority a run
that is cut short has the labels and rainfall before the slow 1-minute sensors.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date, timedelta

from . import sources as S
from .cli import add_common_args, parse_duration, setup
from .client import Http, collection_years, dataset_file_url, iter_day_pages
from .onemap import onemap_search
from .scrapers import parse_telegram_page
from .sinks import FilesApiSink, Sink, read_json, write_json
from .wayback import telegram_wayback

log = logging.getLogger(__name__)

CSV_SOURCES = [s.name for s in S.API_SOURCES if s.csv_collection]


def yesterday() -> date:
    return S.sgt_today() - timedelta(days=1)


def api_start(http: Http, source: S.ApiSource) -> date:
    """First day the API backfill covers: the day after the last published CSV year, else the API's own start."""
    if source.csv_collection:
        years = [
            y
            for y in collection_years(http, source.csv_collection)
            if y >= S.HISTORY_START.year
        ]
        return date(max(years) + 1, 1, 1) if years else S.HISTORY_START
    return max(source.api_start or S.HISTORY_START, S.HISTORY_START)


def api_days(
    http: Http,
    sink: Sink,
    *,
    sources: Sequence[S.ApiSource],
    runner: str,
    start: date | None = None,
    end: date | None = None,
    deadline: float | None = None,
) -> dict[str, dict]:
    """Land every API page of every day in [start, end] (default: each source's start .. yesterday), one gzipped
    JSONL file per source-day. A day whose file exists is skipped; a day without data gets a file holding the 404."""
    end = end or yesterday()
    summary: dict[str, dict] = {}
    for source in sources:
        first = start or api_start(http, source)
        stats = summary[source.name] = {
            "first": first.isoformat(),
            "last": end.isoformat(),
            "landed": 0,
            "empty": 0,
            "skipped": 0,
        }
        log.info("%s: API day pages %s .. %s", source.name, first, end)
        for day in S.days(first, end):
            if deadline is not None and time.monotonic() >= deadline:
                stats["stopped_before"] = day.isoformat()
                log.info("max runtime reached; stopping before %s %s", source.name, day)
                return summary
            path = S.backfill_path(source.name, day)
            if sink.exists(path):
                stats["skipped"] += 1
                continue
            lines, frames = [], 0
            for page, resp in enumerate(
                iter_day_pages(http, source.endpoint, day, source.query), start=1
            ):
                lines.append(
                    S.api_envelope(
                        source.name,
                        resp,
                        page=page,
                        fetched_at=S.utcnow(),
                        runner=runner,
                    )
                )
                if source.name == S.RADAR and resp.status == 200:
                    frames += archive_radar_frames(http, sink, resp.body)[0]
            # written last: its presence marks the day done
            sink.write_bytes(path, S.gzipped(S.jsonl(lines)))
            # no data comes back either as a 404 or as a 200 with no records (e.g. flood alerts before Oct 2025)
            empty = not any(S.record_count(line["body"]) for line in lines)
            stats["empty" if empty else "landed"] += 1
            log.info(
                "%s %s: %s",
                source.name,
                day,
                "no data"
                if empty
                else f"{len(lines)} pages" + (f", {frames} frames" if frames else ""),
            )
    return summary


def archive_radar_frames(
    http: Http, sink: Sink, body: dict, *, after: str = ""
) -> tuple[int, str]:
    """Save the PNG behind each radar record newer than `after`; its presigned URL expires ~25 min after the API
    response, so frames are fetched straight away. Returns (frames saved, newest record timestamp)."""
    records = ((body or {}).get("data") or {}).get("records") or []
    saved, newest = 0, after
    for rec in sorted(records, key=lambda r: r.get("timestamp") or ""):
        ts, url = rec.get("timestamp"), (rec.get("image") or {}).get("url")
        if not ts or not url or ts <= after:
            continue
        sink.write_bytes(S.radar_png_path(ts), http.get(url).content)
        saved, newest = saved + 1, max(newest, ts)
    return saved, newest


def hist_csv(
    http: Http,
    sink: Sink,
    *,
    sources: Sequence[S.ApiSource],
    years: Sequence[int],
    extra: dict[str, dict[int, str]] | None = None,
    gzip_files: bool,
    deadline: float | None = None,
) -> dict[str, dict]:
    """data.gov.sg "Historical ... (YYYY)" CSVs for `years`. Years not published yet are reported, and stay with
    `api_days`; `extra` adds files published outside their collection: {source name: {year: dataset id}}."""
    summary: dict[str, dict] = {}
    for source in sources:
        if not source.csv_collection:
            continue
        published = {
            **collection_years(http, source.csv_collection),
            **(extra or {}).get(source.name, {}),
        }
        stats = summary[source.hist_name] = {
            "landed": [],
            "skipped": [],
            "not_published": [],
        }
        for year in years:
            dataset_id = published.get(year)
            if not dataset_id:
                stats["not_published"].append(year)
                log.info("%s %d: not published on data.gov.sg", source.hist_name, year)
                continue
            path = S.hist_csv_path(source.hist_name, year, dataset_id, gzip_files)
            if sink.exists(path):
                stats["skipped"].append(year)
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return summary
            log.info("%s %d: downloading %s", source.hist_name, year, dataset_id)
            url = dataset_file_url(http, dataset_id, initiate=True)
            size = _download(http, sink, url, path, gzip_files)
            stats["landed"].append(year)
            log.info(
                "%s %d: landed %s (%.0f MB)", source.hist_name, year, path, size / 1e6
            )
    return summary


def _download(http: Http, sink: Sink, url: str, path: str, gz: bool) -> int:
    """Stream a presigned URL into the sink, gzipping on the way if asked; returns the landed size."""
    with http.get(url, stream=True) as resp:
        return sink.write_stream(
            path, resp.iter_content(chunk_size=1 << 20), compress=gz
        )


def telegram_archive(
    http: Http, sink: Sink, *, runner: str, polite_delay: float = 1.5
) -> dict:
    """Every post still visible on t.me/s/pubfloodalerts, paging back with ?before=<oldest id seen>."""
    started = S.utcnow()
    posts: dict[int, dict] = {}
    before, pages = None, 0
    while True:
        url = S.TELEGRAM_URL if before is None else f"{S.TELEGRAM_URL}?before={before}"
        page = parse_telegram_page(http.get(url).text)
        pages += 1
        fresh = [
            p
            for p in page
            if p["message_id"] not in posts
            and (before is None or p["message_id"] < before)
        ]
        for post in fresh:
            posts[post["message_id"]] = {**post, "page_url": url}
        if not fresh or min(p["message_id"] for p in fresh) <= 1:
            break
        before = min(p["message_id"] for p in fresh)
        http.sleep(polite_delay)
    lines = [
        S.record(S.TELEGRAM, fetched_at=started, runner=runner, **posts[i])
        for i in sorted(posts)
    ]
    path = S.run_path(S.TELEGRAM, "archive", started, ext="jsonl.gz")
    sink.write_bytes(path, S.gzipped(S.jsonl(lines)))
    by_year = Counter((p["posted_at"] or "????")[:4] for p in posts.values())
    return {
        "pages": pages,
        "posts": len(posts),
        "path": path,
        "posts_by_year": dict(sorted(by_year.items())),
    }


def uhslc(http: Http, sink: Sink, *, force: bool = False) -> dict:
    """UHSLC research-quality hourly sea level at Tanjong Pagar (station 699), landed as published."""
    state_key = S.state_path("uhslc")
    state = read_json(sink, state_key)
    if state and not force:
        return {"skipped": True, **state}
    started = S.utcnow()
    data = http.get(S.UHSLC_URL).content
    path = S.run_path(S.UHSLC, "download", started, ext="csv")
    sink.write_bytes(path, data)
    state = {
        "path": path,
        "bytes": len(data),
        "rows": data.count(b"\n"),
        "downloaded_at": S.iso(started),
    }
    write_json(sink, state_key, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Historical backfill, 2021 up to yesterday. Every subcommand resumes: it skips what has landed."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    csv = commands.add_parser(
        "csv",
        help="data.gov.sg yearly CSVs (years not published yet are left to api-days)",
    )
    csv.add_argument(
        "--sources", nargs="+", choices=CSV_SOURCES, help="default: all six"
    )
    csv.add_argument(
        "--years",
        type=_years,
        default=_years(f"{S.HISTORY_START.year}-{S.sgt_today().year}"),
    )
    csv.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        metavar="SOURCE:YEAR:ID",
        help="a yearly file published outside its collection, e.g. nea_rainfall:2025:d_abc...",
    )
    csv.add_argument(
        "--gzip",
        choices=("auto", "yes", "no"),
        default="auto",
        help="auto: gzip when uploading (staged under $FLOODSENSE_TMP or ~/.cache)",
    )

    api = commands.add_parser(
        "api-days", help="v2 API day pages after the last CSV year, through yesterday"
    )
    api.add_argument(
        "--sources",
        nargs="+",
        choices=list(S.API_SOURCES_BY_NAME),
        help="default: all, by priority",
    )
    api.add_argument(
        "--start",
        type=date.fromisoformat,
        help="default: per source (after its CSV years)",
    )
    api.add_argument("--end", type=date.fromisoformat, help="default: yesterday (SGT)")
    api.add_argument(
        "--last-days",
        type=int,
        metavar="N",
        help="only the N days up to --end, e.g. 2 for the daily repair",
    )

    commands.add_parser("telegram", help="the whole PUB Flood Alerts Telegram channel")
    wayback = commands.add_parser(
        "telegram-wayback",
        help="posts PUB deleted from the channel, from web archive captures (sgflood, then pubfloodalerts)",
    )
    wayback.add_argument(
        "--until",
        type=date.fromisoformat,
        default=S.WAYBACK_UNTIL,
        help="last capture day to read, UTC (default %(default)s; later captures repeat what `telegram` lands)",
    )
    commands.add_parser(
        "onemap",
        help="OneMap searches for the alert locations in the landed Telegram posts not searched yet (for silver)",
    )
    uhslc_cmd = commands.add_parser(
        "uhslc", help="UHSLC hourly sea level, Tanjong Pagar"
    )
    uhslc_cmd.add_argument(
        "--force", action="store_true", help="download again even if already landed"
    )

    for sub in commands.choices.values():
        add_common_args(sub)
        sub.add_argument(
            "--max-runtime",
            type=parse_duration,
            help="stop cleanly after this long, e.g. 5h30m",
        )
    args = parser.parse_args(argv)
    http, sink = setup(args, share=0.6)
    deadline = time.monotonic() + args.max_runtime if args.max_runtime else None

    if args.command in ("csv", "api-days"):
        chosen = set(args.sources or S.API_SOURCES_BY_NAME)
        sources = [s for s in S.API_SOURCES if s.name in chosen]  # keep priority order
    if args.command == "csv":
        gz = args.gzip == "yes" or (
            args.gzip == "auto" and isinstance(sink, FilesApiSink)
        )
        summary = hist_csv(
            http,
            sink,
            sources=sources,
            years=args.years,
            extra=_extra_ids(args.dataset_id),
            gzip_files=gz,
            deadline=deadline,
        )
    elif args.command == "api-days":
        start = args.start
        if args.last_days:
            start = (args.end or yesterday()) - timedelta(days=args.last_days - 1)
        summary = api_days(
            http,
            sink,
            sources=sources,
            runner=args.runner,
            start=start,
            end=args.end,
            deadline=deadline,
        )
    elif args.command == "telegram":
        summary = telegram_archive(http, sink, runner=args.runner)
    elif args.command == "telegram-wayback":
        summary = telegram_wayback(
            http, sink, runner=args.runner, until=args.until, deadline=deadline
        )
    elif args.command == "onemap":
        summary = onemap_search(http, sink, runner=args.runner, deadline=deadline)
    else:
        summary = uhslc(http, sink, force=args.force)
    print(json.dumps(summary, indent=1))
    return 0


def _years(text: str) -> list[int]:
    years: list[int] = []
    for part in text.split(","):
        first, _, last = part.strip().partition("-")
        years.extend(range(int(first), int(last or first) + 1))
    return years


def _extra_ids(specs: list[str]) -> dict[str, dict[int, str]]:
    extra: dict[str, dict[int, str]] = {}
    for spec in specs:
        name, year, dataset_id = spec.split(":")
        if name not in CSV_SOURCES:
            raise SystemExit(
                f"--dataset-id {spec}: {name} has no CSV history (choose from {CSV_SOURCES})"
            )
        extra.setdefault(name, {})[int(year)] = dataset_id
    return extra
