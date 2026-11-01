"""FloodSense API and web app.

    /api/now                 every area's warning in force and newest rating (ml.predictions), its rain in the last hour,
                             the tide, freshness
    /api/areas/{area}        one area's tiers and chance over the last hours
    /api/locate?q=           a street, building or postal code (OneMap) -> the forecast area it falls in
    /api/track-record        how the live model did since 20 Dec 2025, after its training data ended (ml.scoreboard)
    /api/replays             storm days the model never trained on
    /api/replays/{day}       the champion run tick by tick over that day, with PUB's alerts
    /api/agencies            where the agencies' AI/BI scoreboard and Genie space are, to embed and link
    /api/ask                 a question for the Genie space (POST), answered at /api/ask/{conversation}/{message}
    /api/subscriptions       warnings by email or text: the visitor's (GET), a new one (POST), stop one (DELETE /{id})
    /api/follow              where the Telegram bot is that warns people wherever they are

The React app (frontend/dist) is served from /.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
# a replay reads from gold.area_features the key and every feature the champion reads (Champion.features, from the
# model's own contract), so it scores a past day on the inputs live scoring gives it: the nearby rain and the weather
# included, which a fixed list here once left out


def _t(schema: str, name: str) -> str:
    return S.table(schema, name)


async def _query(sql: str, params: dict | None = None) -> list[dict]:
    try:
        return await asyncio.to_thread(db.query, S.warehouse_id, sql, params)
    except db.QueryError as exc:
        raise HTTPException(503, str(exc)) from exc


# --- now -------------------------------------------------------------------------------------------------------------

# Every Moderate or Elevated rating is a warning for the next hour, so the map shows an area warned until an hour after
# the rating that warned it, and Elevated has priority: the same rule as flood-live's Telegram posts
# (ml/broadcast.in_force) and the replays (frontend lib/warnings.ts).
WARNING = pd.Timedelta(hours=1)
WARNING_HISTORY_H = 6  # how far back a warning's first rating is looked for


def warning_in_force(warned: list[dict], tick: pd.Timestamp) -> dict | None:
    """The warning in force at `tick` from one area's Medium and High ratings, oldest first: High while a High rating
    is under an hour old, else Medium while any is; None when neither. It starts at `since` (its first rating, back
    through ratings at most an hour apart), and its tier holds until an hour after `issued`, the rating that holds it,
    whose chance and reason it carries."""
    stamped = [(pd.Timestamp(r["ts"]), r) for r in warned if pd.Timestamp(r["ts"]) <= tick]
    live = [(t, r) for t, r in stamped if t > tick - WARNING]
    if not live:
        return None
    high = [(t, r) for t, r in live if r["tier"] == "High"]
    issued_at, issued = (high or live)[-1]
    since = stamped[-1][0]
    for t, _ in reversed(stamped[:-1]):
        if since - t > WARNING:
            break
        since = t
    return {
        "tier": "High" if high else "Medium",
        "since": since.isoformat(),
        "issued": issued_at.isoformat(),
        "until": (issued_at + WARNING).isoformat(),
        "p_alert": issued.get("p_alert"),
        "reason": issued.get("reason"),
    }


def _now_sql() -> dict[str, str]:
    p, r, s, t, h = (
        _t("ml", "predictions"),
        _t("silver", "rainfall_5min"),
        _t("gold", "area_rain_stations"),
        _t("gold", "tide_levels"),
        _t("bronze", "ingest_heartbeat"),
    )
    return {
        # the newest tick, and every model that scored it (the pair wins over the rule below), with the warned ratings
        # of the hours before it, which hold the warnings still in force
        "tiers": f"""
            WITH newest AS (SELECT max(ts) AS ts FROM {p})
            SELECT model, area, {p}.ts, tier, p_alert, reason, {p}.ts = newest.ts AS newest FROM {p}, newest
            WHERE {p}.ts = newest.ts OR (tier <> 'Low' AND {p}.ts > newest.ts - INTERVAL {WARNING_HISTORY_H} HOURS)""",
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
    rated, rain, tide, fetched = await asyncio.gather(
        *(_query(sql[k]) for k in ("tiers", "rain", "tide", "fetched"))
    )
    tiers = [{k: v for k, v in row.items() if k != "newest"} for row in rated if row["newest"]]
    models_seen = {row["model"] for row in tiers}
    pair = next(
        (m for m in sorted(models_seen) if not m.startswith("rainfall_rule")), None
    )
    model_name = pair or next(iter(models_seen), None)
    warned: dict[str, list[dict]] = {}
    for row in sorted(rated, key=lambda r: r["ts"]):
        if row["model"] == model_name and row["tier"] != "Low":
            warned.setdefault(row["area"], []).append(row)
    rain_by_area = {row["area"]: row for row in rain}
    areas = []
    for row in tiers:
        if row["model"] != model_name:
            continue
        rained = rain_by_area.get(row["area"], {})
        warning = warning_in_force(warned.get(row["area"], []), pd.Timestamp(row["ts"]))
        areas.append(
            {
                **row,
                "rated": row["tier"],  # the newest rating alone
                "tier": warning["tier"] if warning else "Low",  # the warning in force, which the map shows
                "warning": warning,
                "rain_60min_mm": rained.get("rain_60min_mm"),
            }
        )
    add_nearby(areas)
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
POINTS = {p["area"]: p for p in AREA_POINTS}
# A warning this close is shown beside an area's own tier, never changing it: on the 2026 storms v5 had a Medium in
# force within 5 km of PUB's alert before PUB for 36% of events, against 14% in the alert's own area (23 Oct 2026).
NEARBY_KM = 5.0
TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def nearest_area(lat: float, lon: float) -> str:
    """The forecast area whose label point is nearest, which is how silver places an alert in an area."""
    return min(AREA_POINTS, key=lambda a: distance_km(lat, lon, a["lat"], a["lon"]))["area"]


def add_nearby(areas: list[dict], km: float = NEARBY_KM) -> None:
    """Give each area `nearby`: the other area at Medium or above whose label point lies within `km` of its own, the
    highest tier then the highest chance first ({area, tier, km, p_alert}), or None."""
    warned = [a for a in areas if TIER_RANK.get(a["tier"], 0) > 0 and a["area"] in POINTS]
    for a in areas:
        here, best = POINTS.get(a["area"]), None
        for w in warned if here else ():
            if w["area"] == a["area"]:
                continue
            there = POINTS[w["area"]]
            d = distance_km(here["lat"], here["lon"], there["lat"], there["lon"])
            rank = (TIER_RANK[w["tier"]], w.get("p_alert") or 0.0)
            if d <= km and (best is None or rank > best[0]):
                best = (rank, {"area": w["area"], "tier": w["tier"], "km": round(d, 1), "p_alert": w.get("p_alert")})
        a["nearby"] = best[1] if best else None


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
# How FloodSense did on the storms after its training data ended (from contract.TEST_START, 20 Dec 2025), as
# flood-scoreboard publishes it for the agencies' dashboard and Genie (ml/publish.py): the flash floods it caught, its
# lead time before them, and the chance of a PUB alert or a flash flood given a warning, at Moderate or above and at
# Elevated. Nothing here is cross-validated on the training data. The tiers' cut-offs come from the champion's run.
MEASURES = (
    "flash_floods", "floods_warned", "floods_caught", "floods_caught_share_lo", "floods_caught_share_hi",
    "flood_mean_lead_min", "warnings", "warnings_followed_by_alert", "warnings_followed_by_flood",
)  # fmt: skip


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
    rows = db.query(
        S.warehouse_id,
        f"""SELECT warning_level, model_version, CAST(period_start AS STRING) AS period_start,
                   CAST(period_end AS STRING) AS period_end, {", ".join(MEASURES)}
            FROM {_t("ml", "scoreboard")}
            WHERE period = 'after_training' AND model = 'FloodSense'""",
    )
    by_level = {r["warning_level"]: r for r in rows}
    moderate, elevated = by_level.get("Moderate or above"), by_level.get("Elevated")
    own = [a.alias_name for a in registered.aliases or [] if a.version_num == version]
    return {
        "model": models.version_name(S.model, version, own),
        "version": str(version),
        "trained_at": mv.created_at,
        # the tiers as fixed chances of a PUB alert within the hour (final_1 on)
        "medium_p": m.get("pair_medium_p"),
        "high_p": m.get("pair_high_p"),
        # the scoreboard flood-scoreboard last published, which may be of an earlier version than the champion's
        "test": {
            "model": moderate["model_version"],
            "from": moderate["period_start"],
            "to": moderate["period_end"],
            "moderate": {k: moderate[k] for k in MEASURES},
            "elevated": {k: elevated[k] for k in MEASURES} if elevated else None,
        }
        if moderate
        else None,
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

    async def load():
        try:
            champ = await asyncio.to_thread(models.champion, S.model_full_name, S.model)
        except Exception as exc:
            raise HTTPException(
                503, f"The model couldn't be loaded for the replay: {exc}"
            ) from exc
        columns = ", ".join(("area", "ts", *champ.features))
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


# --- for agencies ----------------------------------------------------------------------------------------------------
# The scoreboard is an AI/BI dashboard the browser embeds: the viewer's own Databricks sign-in shows it, so the app only
# says where it is. Questions go to the Genie space through the Conversation API as the app's service principal (CAN_RUN
# from the app's genie-space resource). Genie takes 10-30 s, longer than a request should wait, so asking starts a
# message and the browser polls for the answer. Both read the tables flood-scoreboard publishes (ml/publish.py).

GENIE_DONE = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}
GENIE_ROWS = 100  # an answer's rows shown; Genie's answers to the space's questions are a few dozen at most
# A first question asked again within this long gets the answer already given: the demo's questions answer at once,
# and a follow-up then continues that conversation.
ASK_REUSE_S = 6 * 3600
_GENIE_ID = r"^[0-9A-Za-z-]{8,64}$"


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=400)
    conversation_id: str | None = Field(default=None, pattern=_GENIE_ID)


def _agency_links() -> dict:
    w = db.client()
    host = w.config.host.rstrip("/")
    host = host if host.startswith("http") else f"https://{host}"
    workspace = os.environ.get("DATABRICKS_WORKSPACE_ID") or str(w.get_workspace_id())
    return {
        "dashboard": {
            "embed_url": f"{host}/embed/dashboardsv3/{S.dashboard}?o={workspace}",
            "url": f"{host}/dashboardsv3/{S.dashboard}/published?o={workspace}",
        }
        if S.dashboard
        else None,
        "genie": {"url": f"{host}/genie/rooms/{S.genie_space}?o={workspace}"} if S.genie_space else None,
    }


@app.get("/api/agencies")
async def agencies() -> dict:
    try:
        return await asyncio.to_thread(db.cached, "agencies", 24 * 3600, _agency_links)
    except Exception as exc:
        raise HTTPException(503, f"The agencies' scoreboard isn't reachable: {exc}") from exc


def _genie(method: str, path: str, body: dict | None = None) -> dict:
    """One Conversation API call on the agencies' Genie space."""
    if not S.genie_space:
        raise db.QueryError("no Genie space is configured (FLOODSENSE_GENIE_SPACE)")
    return db.client().api_client.do(method, f"/api/2.0/genie/spaces/{S.genie_space}{path}", body=body)


@app.post("/api/ask")
async def ask(q: Question) -> dict:
    """Start Genie on a question, a new conversation or a follow-up in `conversation_id`: {conversation_id,
    message_id}, which /api/ask/{conversation_id}/{message_id} answers."""
    text = " ".join(q.question.split())
    key = f"ask:{text.lower()}"
    if q.conversation_id is None:
        hit = db.fresh(key, ASK_REUSE_S)
        if hit and db.fresh(f"answer:{hit['conversation_id']}/{hit['message_id']}", ASK_REUSE_S) is not None:
            return hit
    try:
        if q.conversation_id:
            m = await asyncio.to_thread(
                _genie, "POST", f"/conversations/{q.conversation_id}/messages", {"content": text}
            )
            started = {"conversation_id": q.conversation_id, "message_id": m.get("message_id") or m["id"]}
        else:
            r = await asyncio.to_thread(_genie, "POST", "/start-conversation", {"content": text})
            started = {"conversation_id": r["conversation_id"], "message_id": r["message_id"]}
    except Exception as exc:
        raise HTTPException(503, f"Genie isn't reachable right now: {exc}") from exc
    if q.conversation_id is None:
        db.put(key, started)
    return started


def _answer(conversation_id: str, message_id: str) -> dict:
    """Genie's message as the browser shows it: where it is, then its text, the SQL behind it, the first GENIE_ROWS
    rows, and the follow-ups Genie suggests."""
    base = f"/conversations/{conversation_id}/messages/{message_id}"
    m = _genie("GET", base)
    status = m.get("status") or "SUBMITTED"
    out = {
        "status": status,
        "done": status in GENIE_DONE,
        "text": None,
        "sql": None,
        "description": None,
        "columns": [],
        "rows": [],
        "row_count": None,
        "suggested": [],
        "error": (m.get("error") or {}).get("error"),
    }
    if not out["done"]:
        return out
    texts = []
    for a in m.get("attachments") or []:
        if a.get("text"):
            texts.append(a["text"].get("content") or "")
        if a.get("query") and out["sql"] is None:
            out["sql"], out["description"] = a["query"].get("query"), a["query"].get("description")
            if status == "COMPLETED":
                result = _genie("GET", f"{base}/attachments/{a['attachment_id']}/query-result")
                statement = result.get("statement_response") or {}
                manifest = statement.get("manifest") or {}
                rows = (statement.get("result") or {}).get("data_array") or []
                out["columns"] = [c["name"] for c in (manifest.get("schema") or {}).get("columns") or []]
                out["rows"] = rows[:GENIE_ROWS]
                out["row_count"] = manifest.get("total_row_count", len(rows))
        if a.get("suggested_questions"):
            out["suggested"] = a["suggested_questions"].get("questions") or []
    out["text"] = "\n\n".join(t for t in texts if t) or None
    return out


@app.get("/api/ask/{conversation_id}/{message_id}")
async def answer(
    conversation_id: str = PathParam(pattern=_GENIE_ID), message_id: str = PathParam(pattern=_GENIE_ID)
) -> dict:
    key = f"answer:{conversation_id}/{message_id}"
    hit = db.fresh(key, ASK_REUSE_S)
    if hit is not None:
        return hit
    try:
        out = await asyncio.to_thread(_answer, conversation_id, message_id)
    except Exception as exc:
        raise HTTPException(503, f"Genie's answer isn't reachable right now: {exc}") from exc
    # a finished answer never changes; keep it, so a question asked again answers at once
    return db.put(key, out) if out["status"] == "COMPLETED" else out


# --- warnings by email or text ---------------------------------------------------------------------------------------
# The Now view's sign-up. The app only writes the request, as its service principal, to the app schema's
# warning_subscriptions (personal data, kept apart from the analytics schemas). flood-live's broadcast step sends the
# welcome message and every warning after it, and records on the row how the welcome went (ml/broadcast.py); its first
# run creates the table. Addresses go back to the browser masked, since the app is shown on shared screens.

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SG_MOBILE = re.compile(r"^(?:\+?65)?([89]\d{7})$")  # a Singapore mobile number, with or without +65
MAX_AREAS = 10
MAX_ACTIVE = 5  # per signed-in user, so a typo can't fan out


class NewSubscription(BaseModel):
    channel: Literal["email", "sms"]
    address: str = Field(min_length=5, max_length=254)
    areas: list[str] = Field(min_length=1, max_length=MAX_AREAS)
    places: list[str] = Field(default_factory=list, max_length=MAX_AREAS)


def _subscriptions() -> str:
    return _t("app", "warning_subscriptions")


def _visitor(request: Request) -> str:
    """The signed-in Databricks user, which Databricks Apps passes in a header; empty when run locally."""
    return (request.headers.get("x-forwarded-email") or "").strip().lower()


def normalized_address(channel: str, raw: str) -> str:
    """An email address in lower case, or a mobile number as +65 and 8 digits; ValueError if it isn't one."""
    text = raw.strip()
    if channel == "email":
        if not EMAIL.fullmatch(text):
            raise ValueError("That doesn't look like an email address.")
        return text.lower()
    found = SG_MOBILE.fullmatch(re.sub(r"[\s()-]", "", text))
    if not found:
        raise ValueError("Enter a Singapore mobile number: 8 digits starting with 8 or 9.")
    return f"+65{found.group(1)}"


def masked(channel: str, address: str) -> str:
    """'s•••@gmail.com' or '+65 •••• 4567'."""
    if channel == "email":
        name, _, domain = address.partition("@")
        return f"{name[:1]}•••@{domain}"
    return f"+65 •••• {address[-4:]}"


def _shown(row: dict) -> dict:
    return {
        "id": row["id"],
        "channel": row["channel"],
        "address": masked(row["channel"], row["address"]),
        "areas": json.loads(row["areas"] or "[]"),
        "places": json.loads(row["places"] or "[]"),
        "created_at": row["created_at"],
        "welcome": row["welcome"],
        "welcomed_at": row["welcomed_at"],
    }


async def _subscription_query(sql: str, params: dict) -> list[dict]:
    try:
        return await asyncio.to_thread(db.query, S.warehouse_id, sql, params)
    except db.QueryError as exc:
        text = str(exc)
        if "TABLE_OR_VIEW_NOT_FOUND" in text or "SCHEMA_NOT_FOUND" in text:
            raise HTTPException(503, "Warnings by email or text aren't switched on yet. Try again later.") from exc
        raise HTTPException(503, f"Your warnings couldn't be saved or read right now: {text}") from exc


@app.get("/api/subscriptions")
async def my_subscriptions(request: Request, ids: str = Query("", max_length=800)) -> dict:
    """The visitor's active subscriptions: those made while signed in as them, and those this browser remembers."""
    known = [i for i in ids.split(",") if re.fullmatch(r"[0-9a-f-]{36}", i)][:20]
    rows = await _subscription_query(
        f"""SELECT id, channel, address, to_json(areas) AS areas, to_json(places) AS places, created_at, welcome,
                   welcomed_at
            FROM {_subscriptions()}
            WHERE active AND (array_contains(from_json(:ids, 'array<string>'), id)
                              OR (:me <> '' AND created_by = :me))
            ORDER BY created_at""",
        {"ids": json.dumps(known), "me": _visitor(request)},
    )
    return {"subscriptions": [_shown(r) for r in rows]}


@app.post("/api/subscriptions")
async def subscribe(request: Request, new: NewSubscription) -> dict:
    try:
        address = normalized_address(new.channel, new.address)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    areas = sorted({a for a in new.areas if a in POINTS})
    if not areas:
        raise HTTPException(422, "Pick at least one of the 47 forecast areas.")
    me = _visitor(request)
    if me:
        mine = await _subscription_query(
            f"SELECT count(*) AS n FROM {_subscriptions()} WHERE active AND created_by = :me", {"me": me}
        )
        if mine and mine[0]["n"] >= MAX_ACTIVE:
            raise HTTPException(422, f"You have {MAX_ACTIVE} subscriptions already. Stop one first.")
    row = {
        "id": str(uuid.uuid4()),
        "channel": new.channel,
        "address": address,
        "areas": json.dumps(areas),
        "places": json.dumps([p.strip()[:80] for p in new.places if p.strip()][:MAX_AREAS]),
    }
    await _subscription_query(
        f"""INSERT INTO {_subscriptions()} (id, created_at, created_by, channel, address, areas, places, active)
            VALUES (:id, current_timestamp(), nullif(:me, ''), :channel, :address,
                    from_json(:areas, 'array<string>'), from_json(:places, 'array<string>'), true)""",
        {**row, "me": me},
    )
    return _shown({**row, "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "welcome": None, "welcomed_at": None})


@app.delete("/api/subscriptions/{subscription_id}")
async def unsubscribe(subscription_id: str = PathParam(pattern=r"^[0-9a-f-]{36}$")) -> dict:
    await _subscription_query(
        f"UPDATE {_subscriptions()} SET active = false, stopped_at = current_timestamp() WHERE id = :id AND active",
        {"id": subscription_id},
    )
    return {"stopped": subscription_id}


# Following a phone's location needs an app on the phone, so it is Telegram's: the visitor shares their live location
# with the bot, and flood-live answers them and tells them the warning where they are (ml/follow.py). The app only links
# to the bot, and never sees where anyone is.


@app.get("/api/follow")
async def follow() -> dict:
    bot = S.telegram_bot
    return {"bot": bot or None, "url": f"https://t.me/{bot}?start=follow" if bot else None}


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
