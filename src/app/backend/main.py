"""FloodSense API and web app.

    /api/now                 every area's newest tier (ml.predictions), its rain in the last hour, the tide, freshness
    /api/areas/{area}        one area's tiers and chance over the last hours
    /api/locate?q=           a street, building or postal code (OneMap) -> the forecast area it falls in
    /api/track-record        how the live model did out of fold, from its MLflow run
    /api/replays             storm days the model never trained on
    /api/replays/{day}       the champion run tick by tick over that day, with PUB's alerts

The React app (frontend/dist) is served from /.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db
from . import model as models
from .settings import load

S = load()
app = FastAPI(title="FloodSense", docs_url="/api/docs", openapi_url="/api/openapi.json")
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
AREAS_GEOJSON = (
    Path(__file__).resolve().parents[1] / "frontend" / "public" / "areas.geojson"
)

# the model was trained on PUB alerts up to 19 Dec 2025 (contract.TEST_START is 20 Dec); replays start after it
REPLAY_FROM = date(2025, 12, 20)
TIERS = ("Low", "Medium", "High")
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
# what a replay reads from gold.area_features: the key and every model feature
REPLAY_COLUMNS = (
    "area", "ts", "rain_15min_mm", "rain_30min_mm", "rain_60min_mm", "rain_gauges", "forecast_rain_class",
    "forecast_age_min", "tide_height_m", "minutes_to_high_tide",
)  # fmt: skip


def _t(schema: str, name: str) -> str:
    return S.table(schema, name)


async def _query(sql: str, params: dict | None = None) -> list[dict]:
    try:
        return await asyncio.to_thread(db.query, S.warehouse_id, sql, params)
    except db.QueryError as exc:
        raise HTTPException(503, str(exc)) from exc


# --- now -------------------------------------------------------------------------------------------------------------


def _now_sql() -> dict[str, str]:
    p, r, s, t, h = (
        _t("ml", "predictions"),
        _t("silver", "rainfall_5min"),
        _t("gold", "area_rain_stations"),
        _t("gold", "tide_levels"),
        _t("bronze", "ingest_heartbeat"),
    )
    return {
        # the newest tick, and every model that scored it; the pair wins over the rule below
        "tiers": f"""
            SELECT model, area, ts, tier, p_alert, reason FROM {p}
            WHERE ts = (SELECT max(ts) FROM {p})""",
        # the heaviest hour of rain among the gauges within 5 km of each area, ending at the newest reading
        "rain": f"""
            WITH newest AS (SELECT max(ts) AS ts FROM {r}),
            per_gauge AS (
              SELECT station_id, sum(rain_mm) AS mm FROM {r}, newest
              WHERE {r}.ts > newest.ts - INTERVAL 60 MINUTES AND {r}.ts <= newest.ts GROUP BY station_id)
            SELECT a.area, max(g.mm) AS rain_60min_mm, (SELECT ts FROM newest) AS as_of
            FROM {s} a JOIN per_gauge g USING (station_id) GROUP BY a.area""",
        "tide": f"""
            SELECT ts, predicted_m, next_high_ts, next_high_m, minutes_to_high_tide FROM {t}
            WHERE ts BETWEEN current_timestamp() - INTERVAL 6 HOURS AND current_timestamp() + INTERVAL 18 HOURS
            ORDER BY ts""",
        "fetched": f"SELECT max(fetched_at) AS fetched_at FROM {h}",
    }


async def _load_now() -> dict:
    sql = _now_sql()
    tiers, rain, tide, fetched = await asyncio.gather(
        *(_query(sql[k]) for k in ("tiers", "rain", "tide", "fetched"))
    )
    models_seen = {row["model"] for row in tiers}
    pair = next(
        (m for m in sorted(models_seen) if not m.startswith("rainfall_rule")), None
    )
    model_name = pair or next(iter(models_seen), None)
    rain_by_area = {row["area"]: row for row in rain}
    areas = []
    for row in tiers:
        if row["model"] != model_name:
            continue
        rained = rain_by_area.get(row["area"], {})
        areas.append({**row, "rain_60min_mm": rained.get("rain_60min_mm")})
    # the tide every 30 minutes for the chart, and the tick nearest now
    now_ts = pd.Timestamp.now(tz="UTC")
    curve = [row for i, row in enumerate(tide) if i % 6 == 0]
    current = min(tide, key=lambda r: abs(pd.Timestamp(r["ts"]) - now_ts), default=None)
    return {
        "tick": tiers[0]["ts"] if tiers else None,
        "model": model_name,
        "areas": sorted(areas, key=lambda a: a["area"]),
        "rain_as_of": rain[0]["as_of"] if rain else None,
        "fetched_at": fetched[0]["fetched_at"] if fetched else None,
        "tide": {"now": current, "curve": curve},
    }


@app.get("/api/now")
async def now() -> dict:
    """Cached for a minute: a tick is 5 minutes, and every visitor sees the same map."""
    try:
        return await _cached("now", 60, _load_now)
    except HTTPException:
        stale = db.stale("now")
        if stale is None:
            raise
        return {**stale, "stale": True}


async def _cached(key: str, ttl: float, load):
    """`load()`'s value, reused for `ttl` seconds across requests."""
    hit = db.fresh(key, ttl)
    if hit is not None:
        return hit
    return db.put(key, await load())


@app.get("/api/areas/{area}")
async def area_history(area: str, hours: int = Query(6, ge=1, le=48)) -> dict:
    rows = await _query(
        f"""SELECT ts, model, tier, p_alert, reason FROM {_t("ml", "predictions")}
            WHERE area = :area AND ts >= current_timestamp() - make_interval(0, 0, 0, 0, :hours, 0, 0)
            ORDER BY ts""",
        {"area": area, "hours": str(hours)},
    )
    return {"area": area, "points": rows}


# --- where am I ------------------------------------------------------------------------------------------------------


def _areas() -> list[dict]:
    import json

    features = json.loads(AREAS_GEOJSON.read_text(encoding="utf-8"))["features"]
    return [f["properties"] for f in features]


AREA_POINTS = _areas()


def nearest_area(lat: float, lon: float) -> str:
    """The forecast area whose label point is nearest, which is how silver places an alert in an area."""

    def km(a: dict) -> float:
        p1, p2 = math.radians(lat), math.radians(a["lat"])
        h = (
            math.sin((p2 - p1) / 2) ** 2
            + math.cos(p1)
            * math.cos(p2)
            * math.sin(math.radians(a["lon"] - lon) / 2) ** 2
        )
        return 2 * 6371.0088 * math.asin(math.sqrt(h))

    return min(AREA_POINTS, key=km)["area"]


@app.get("/api/locate")
async def locate(q: str = Query(..., min_length=2, max_length=80)) -> dict:
    try:
        async with httpx.AsyncClient(timeout=8) as http:
            resp = await http.get(
                ONEMAP,
                params={
                    "searchVal": q,
                    "returnGeom": "Y",
                    "getAddrDetails": "Y",
                    "pageNum": 1,
                },
            )
        found = resp.json().get("results") or []
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            502,
            "Address search isn't reachable right now. Pick your area on the map instead.",
        ) from exc
    if not found:
        raise HTTPException(
            404,
            f'No place in Singapore matches "{q}". Try a street name, building or postal code.',
        )
    top = found[0]
    lat, lon = float(top["LATITUDE"]), float(top["LONGITUDE"])
    return {
        "query": q,
        "place": top.get("ADDRESS") or top.get("SEARCHVAL"),
        "lat": lat,
        "lon": lon,
        "area": nearest_area(lat, lon),
    }


@app.get("/api/nearest")
async def nearest(
    lat: float = Query(..., ge=1.1, le=1.5), lon: float = Query(..., ge=103.5, le=104.2)
) -> dict:
    return {"area": nearest_area(lat, lon)}


# --- track record ----------------------------------------------------------------------------------------------------


def _load_track_record() -> dict:
    w = db.client()
    registered = w.registered_models.get(S.model_full_name, include_aliases=True)
    version = next(
        (a.version_num for a in registered.aliases or [] if a.alias_name == "champion"),
        None,
    )
    if version is None:
        raise db.QueryError("no @champion model is registered yet")
    mv = w.model_versions.get(S.model_full_name, version)
    run = w.experiments.get_run(mv.run_id).run
    m = {x.key: x.value for x in run.data.metrics or []}
    p = {x.key: x.value for x in run.data.params or []}
    keep = (
        "oof_alert_events", "oof_alert_days", "oof_alert_flagged_before_pub", "oof_alert_flagged_before_pub_lo",
        "oof_alert_flagged_before_pub_hi", "oof_alert_mean_lead_min", "oof_alert_mean_warning_min_lo",
        "oof_alert_mean_warning_min_hi", "oof_false_alarms_per_area_month", "oof_warned_hours_per_area_month",
        "rule_dev_alert_flagged_before_pub", "rule_dev_alert_mean_lead_min", "rule_dev_false_alarms_per_area_month",
        "without_tide_delta_alert_flagged_before_pub", "without_tide_alert_flagged_only_with",
        "without_tide_alert_flagged_only_without", "without_tide_alert_mcnemar_p",
    )  # fmt: skip
    return {
        "model": f"{S.model} v{version}",
        "version": str(version),
        "trained_at": mv.created_at,
        "development": p.get("development"),
        "rule": p.get("rule"),
        "metrics": {k: m[k] for k in keep if k in m},
    }


@app.get("/api/track-record")
async def track_record() -> dict:
    try:
        return await asyncio.to_thread(db.cached, "track", 3600, _load_track_record)
    except Exception as exc:
        raise HTTPException(
            503, f"The model's track record isn't available: {exc}"
        ) from exc


# --- replays ---------------------------------------------------------------------------------------------------------


@app.get("/api/replays")
async def replays() -> dict:
    alerts, features = _t("silver", "area_alerts"), _t("gold", "area_features")

    async def load():
        rows = await _query(
            f"""SELECT date_format(from_utc_timestamp(alert_ts, 'Asia/Singapore'), 'yyyy-MM-dd') AS day,
                       count(*) AS alerts, count_if(kind = 'flash_flood') AS floods, count(DISTINCT area) AS areas,
                       array_join(slice(array_sort(collect_set(area)), 1, 4), ', ') AS some_areas
                FROM {alerts}
                WHERE alert_ts >= to_utc_timestamp(:start, 'Asia/Singapore')
                  AND alert_ts < (SELECT max(ts) FROM {features})
                GROUP BY 1 ORDER BY floods DESC, alerts DESC LIMIT 8""",
            {"start": REPLAY_FROM.isoformat()},
        )
        return {"from": REPLAY_FROM.isoformat(), "days": rows}

    return await _cached("replays", 3600, load)


def _score_day(day: str, rows: list[dict], alerts: list[dict]) -> dict:
    champ = models.champion(S.model_full_name, S.model)
    start = pd.Timestamp(day).tz_localize("Asia/Singapore")
    ticks = pd.date_range(
        start, start + pd.Timedelta(days=1), freq="5min", inclusive="left"
    )
    names = [a["area"] for a in AREA_POINTS]
    grid = pd.MultiIndex.from_product([names, ticks], names=["area", "ts"]).to_frame(
        index=False
    )
    found = pd.DataFrame(rows)
    if not found.empty:
        found["ts"] = pd.to_datetime(found["ts"], utc=True).dt.tz_convert(
            "Asia/Singapore"
        )
        grid = grid.merge(found, on=["area", "ts"], how="left")
    for column in champ.features:
        if column not in grid:
            grid[column] = float("nan")
    frame = grid.assign(ts=grid["ts"].dt.tz_localize(None))  # the model reads naive SGT
    preds = champ.predict(frame)
    code = preds["tier"].map({"Low": "L", "Medium": "M", "High": "H"}).fillna("L")
    tiers = {}
    first_warned = {}
    for area, part in pd.DataFrame({"area": frame["area"], "code": code}).groupby(
        "area", sort=False
    ):
        tiers[area] = "".join(part["code"])
        hit = (
            part["code"].ne("L").to_numpy().argmax()
            if part["code"].ne("L").any()
            else None
        )
        if hit is not None:
            first_warned[area] = ticks[hit].isoformat()
    return {
        "day": day,
        "model": champ.name,
        "ticks": [t.isoformat() for t in ticks],
        "tiers": tiers,
        "first_warned": first_warned,
        "alerts": alerts,
        "areas_with_features": int(found["area"].nunique()) if not found.empty else 0,
    }


@app.get("/api/replays/{day}")
async def replay(day: str) -> dict:
    if not _DAY.fullmatch(day) or date.fromisoformat(day) < REPLAY_FROM:
        raise HTTPException(
            400,
            f"Pick a day from {REPLAY_FROM} on: the model trained on earlier storms.",
        )
    nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    window = {"day": day, "next": nxt}
    columns = ", ".join(REPLAY_COLUMNS)

    async def load():
        rows, alerts = await asyncio.gather(
            _query(
                f"""SELECT {columns} FROM {_t("gold", "area_features")}
                    WHERE ts >= to_utc_timestamp(:day, 'Asia/Singapore') AND ts < to_utc_timestamp(:next, 'Asia/Singapore')""",
                window,
            ),
            _query(
                f"""SELECT area, alert_ts AS ts, kind, location FROM {_t("silver", "area_alerts")}
                    WHERE alert_ts >= to_utc_timestamp(:day, 'Asia/Singapore')
                      AND alert_ts < to_utc_timestamp(:next, 'Asia/Singapore') ORDER BY alert_ts""",
                window,
            ),
        )
        try:
            return await asyncio.to_thread(_score_day, day, rows, alerts)
        except Exception as exc:
            raise HTTPException(
                503, f"The model couldn't be loaded for the replay: {exc}"
            ) from exc

    return await _cached(f"replay:{day}", 24 * 3600, load)


# --- the web app -----------------------------------------------------------------------------------------------------

if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
async def spa(path: str):
    target = DIST / path
    if path and target.is_file():
        return FileResponse(target)
    index = DIST / "index.html"
    if not index.exists():
        raise HTTPException(
            404, "The web app isn't built: run `npm run build` in src/app/frontend."
        )
    return FileResponse(index)
