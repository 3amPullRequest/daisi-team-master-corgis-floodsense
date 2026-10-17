"""What we ingest and where it lands.

Contract with the pipeline: files under `landing/<name>/` are streamed into `bronze.<name>`, so a source's `name`
is both its landing folder and its bronze table (tests/test_bronze_contract.py checks the two sides agree).

This module is the catalogue every fetcher shares. It fetches nothing itself; it says what exists, where a fetch of it
lands, and what a landed line looks like:

- The sources. API_SOURCES are the nine data.gov.sg v2 real-time endpoints (PUB flood alerts, NEA rainfall, 2-hour
  forecast, wind direction and speed, lightning, air temperature, relative humidity, radar), listed in backfill
  priority: the labels and rainfall first, the slow 1-minute sensors last. Six of them also have a yearly "Historical
  ..." CSV collection (`csv_collection`), landed as `<name>_hist`; the API then only has to cover the years after the
  last published CSV. The rest start where the API does (`api_start`). STATIC_LAYERS are one-off datasets, and the
  TELEGRAM, TIDES, UHSLC, ONEMAP and HEARTBEAT names are the folders that don't come from the real-time API.
- The channel's history. PUB renamed its Telegram channel but kept the post ids, so TELEGRAM_ALIASES says which
  username to ask the web archive for on which days, and WAYBACK_UNTIL where captures stop adding posts the channel
  doesn't still show.
- The landing layout, as path builders. A fetch always lands as a new file and nothing is overwritten in place: a live
  tick as `<name>/live/date=<SGT day>/<name>_<UTC stamp>.jsonl`, a backfilled day as one `<name>/backfill/date=<day>/`
  file whose existence marks the day done, a yearly CSV under `year=`, a static layer under `version=`. Fetcher
  cursors sit in `_state/`, outside every table folder, so Auto Loader never reads them.
- The shape of a landed line. An API page is wrapped untouched in an envelope (`api_envelope`: source, fetched_at,
  runner, the request, the status, the body as received), so bronze keeps exactly what the API said and silver does
  all the interpreting. Every other source gets the same envelope fields plus its own (`record`).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from .client import ApiResponse

SGT = ZoneInfo("Asia/Singapore")

# Earliest day we backfill. PUB's Telegram channel, the only historical label source, starts 2021-10-26.
HISTORY_START = date(2021, 1, 1)


@dataclass(frozen=True)
class ApiSource:
    """A data.gov.sg v2 real-time endpoint, fetched live every tick and backfilled day by day."""

    name: str
    endpoint: str
    params: tuple[tuple[str, str], ...] = ()
    csv_collection: int | None = (
        None  # yearly "Historical ..." CSV collection with the same readings
    )
    api_start: date | None = (
        None  # first day the API serves, for sources without CSV history
    )

    @property
    def query(self) -> dict[str, str]:
        return dict(self.params)

    @property
    def hist_name(self) -> str:
        return f"{self.name}_hist"


# Ordered by backfill priority: labels and rainfall first, slow 1-minute sensors last.
API_SOURCES: tuple[ApiSource, ...] = (
    ApiSource("pub_flood_alerts", "weather/flood-alerts", api_start=date(2025, 9, 1)),
    ApiSource("nea_rainfall", "rainfall", csv_collection=2279),
    ApiSource("nea_two_hr_forecast", "two-hr-forecast", csv_collection=2179),
    ApiSource("nea_wind_direction", "wind-direction", csv_collection=2281),
    ApiSource("nea_wind_speed", "wind-speed", csv_collection=2280),
    ApiSource(
        "nea_lightning",
        "weather",
        params=(("api", "lightning"),),
        api_start=date(2025, 2, 1),
    ),
    ApiSource("nea_air_temperature", "air-temperature", csv_collection=2246),
    ApiSource("nea_relative_humidity", "relative-humidity", csv_collection=2278),
    ApiSource(
        "nea_radar_70km", "weather-radar-images/70km", api_start=date(2026, 7, 1)
    ),
)
API_SOURCES_BY_NAME = {s.name: s for s in API_SOURCES}
RADAR = "nea_radar_70km"


@dataclass(frozen=True)
class StaticLayer:
    name: str
    dataset_id: str
    kind: str  # "geojson" | "csv" | "xlsx"


STATIC_LAYERS: tuple[StaticLayer, ...] = (
    StaticLayer(
        "static_mp2019_land_use", "d_90d86daa5bfaa371668b84fa5f01424f", "geojson"
    ),
    StaticLayer(
        "static_mp2025_land_use", "d_a8c3546b26712e35021f3a681d0353ae", "geojson"
    ),
    StaticLayer(
        "static_pub_water_level_sensors", "d_31333fa5cf0834f012d840365b336610", "xlsx"
    ),
    StaticLayer(
        "static_flood_prone_area_ha", "d_c4aed98f1533eb3a66f65dbb1a30da46", "csv"
    ),
    StaticLayer(
        "static_sla_vertical_control", "d_c1f2b991f3932fb69b48edcceef54d13", "geojson"
    ),
)

TELEGRAM = "pub_telegram_posts"
TELEGRAM_CHANNEL = "pubfloodalerts"
TELEGRAM_URL = f"https://t.me/s/{TELEGRAM_CHANNEL}"


@dataclass(frozen=True)
class TelegramAlias:
    """A username PUB's channel has had, and the days (UTC) when web archive captures under it show PUB's posts."""

    username: str
    first: date
    last: date | None = None  # None: still PUB's


# Post ids survive a username change, so sgflood/140 and pubfloodalerts/140 are the same post. The channel moved to
# pubfloodalerts on 14 Nov 2023, titled "Gov.sg-PUB Flood Alerts" until 3 Apr 2024. An unrelated channel later took
# the name sgflood, and its post ids start again at 1.
TELEGRAM_ALIASES = (
    TelegramAlias("sgflood", date(2021, 10, 26), date(2023, 11, 14)),
    TelegramAlias(TELEGRAM_CHANNEL, date(2023, 11, 14)),
)
# PUB wiped the channel between Apr and Sep 2024 and again around 23 Mar 2025. Posts from 19 Mar 2025 on (id 1111
# up) are still on the channel, so later captures only repeat what `backfill.py telegram` lands.
WAYBACK_UNTIL = date(2025, 3, 31)

TIDES = "nea_tide_timings"
NEA_TIDES_URL = "https://www.nea.gov.sg/corporate-functions/weather/tide-timings"

UHSLC = "uhslc_tanjong_pagar_hourly"
UHSLC_URL = "https://uhslc.soest.hawaii.edu/data/csv/rqds/pacific/hourly/h699a.csv"  # research quality, 1984-2023

ONEMAP = (
    "onemap_search"  # OneMap answers for PUB alert locations, for silver's gazetteer
)
ONEMAP_SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"

HEARTBEAT = "ingest_heartbeat"
RADAR_PNG = "radar_png"  # image archive; no bronze table
STATE = "_state"  # fetcher cursors; outside every table folder, so Auto Loader never reads it


def bronze_landing_names() -> set[str]:
    """Every landing folder that has a bronze table of the same name."""
    names = {s.name for s in API_SOURCES}
    names |= {s.hist_name for s in API_SOURCES if s.csv_collection}
    names |= {layer.name for layer in STATIC_LAYERS}
    return names | {TELEGRAM, TIDES, UHSLC, ONEMAP, HEARTBEAT}


# --- time and paths ------------------------------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stamp(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def sgt_today(now: datetime | None = None) -> date:
    return (now or utcnow()).astimezone(SGT).date()


def days(first: date, last: date) -> Iterable[date]:
    for n in range((last - first).days + 1):
        yield first + timedelta(days=n)


def run_path(name: str, kind: str, ts: datetime, ext: str = "jsonl") -> str:
    """One file per fetch, partitioned by SGT day: `<name>/<kind>/date=YYYY-MM-DD/<name>_<UTC stamp>.<ext>`."""
    return f"{name}/{kind}/date={sgt_today(ts).isoformat()}/{name}_{stamp(ts)}.{ext}"


def backfill_path(name: str, day: date) -> str:
    """All API pages of one SGT day; its existence marks the day as done."""
    return f"{name}/backfill/date={day.isoformat()}/{name}_{day.isoformat()}.jsonl.gz"


def hist_csv_path(name: str, year: int, dataset_id: str, gz: bool) -> str:
    return f"{name}/year={year}/{dataset_id}.csv{'.gz' if gz else ''}"


def static_folder(name: str, version: str) -> str:
    return f"{name}/version={version}"


def wayback_path(timestamp: str, digest: str) -> str:
    """Posts from one web archive capture (its 14-digit UTC timestamp and content digest); its existence marks the
    capture as read."""
    return f"{TELEGRAM}/wayback/{TELEGRAM}_{timestamp}_{digest}.jsonl.gz"


def radar_png_path(record_timestamp: str) -> str:
    """'2026-10-10T18:50:00+08:00' -> radar_png/70km/2026/10/10/rain_70km_202610101850.png (SGT, like NEA's own names)."""
    ts = datetime.fromisoformat(record_timestamp).astimezone(SGT)
    return f"{RADAR_PNG}/70km/{ts:%Y/%m/%d}/rain_70km_{ts:%Y%m%d%H%M}.png"


def state_path(key: str) -> str:
    return f"{STATE}/{key}.json"


# --- landed records --------------------------------------------------------------------------------------------------


def api_envelope(
    source: str, resp: ApiResponse, *, page: int, fetched_at: datetime, runner: str
) -> dict:
    """One landed line: the API body exactly as received, plus how and when it was fetched."""
    return {
        "source": source,
        "fetched_at": iso(fetched_at),
        "runner": runner,
        "request": {"url": resp.url, "params": resp.params, "page": page},
        "status": resp.status,
        "body": resp.body,
    }


def record(source: str, *, fetched_at: datetime, runner: str, **fields) -> dict:
    """A landed line for non-API sources (Telegram posts, tide rows, static features, heartbeats)."""
    return {"source": source, "fetched_at": iso(fetched_at), "runner": runner, **fields}


def jsonl(records: Iterable[dict]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records
    ).encode()


def gzipped(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def record_count(body: object) -> int:
    """Readings, forecast items or records in a v2 API body."""
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, dict):
        for key in ("readings", "items", "records"):
            if isinstance(data.get(key), list):
                return len(data[key])
    return 0
