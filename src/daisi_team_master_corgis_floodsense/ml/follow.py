"""Warnings wherever you are: FloodSense's Telegram bot follows a person's live location, and messages them when the
area they're in is warned.

Someone opens the bot (t.me/<bot>?start=follow, linked from the app's Now view), taps Start, and shares their live
location (📎, Location, Share My Live Location, for a few hours or "until I turn it off"), or sends a pin to follow one
place. Telegram sends each new position of a live location as an edit of that message, and an edit with no
`live_period` when the sharing stops. Nothing outside can reach flood-live, so the bot fetches its own messages: after
broadcast.run, each run's score task reads the updates Telegram kept since the last run (getUpdates, kept up to 24
hours), answers them, and confirms them by asking once more past the last one. So the bot answers within one run.

Only the area is kept, never the coordinates: <app schema>.telegram_followers holds each chat's forecast area (the one
the position falls in, as the app's map draws the areas: NEA's around their label points, clipped to Singapore's land,
so a phone in Johor or out at sea is in none), whether it follows a live location or a pin, until when, and the last
warning the chat was told. A follower hears when a warning starts, steps up or steps down in their area, or when they
move into an area under a warning they haven't heard about; nothing while it holds, when it lapses, or for a tick more
than broadcast.FRESH old. A live location that ends stops the following, with one message to say so. /stop stops it
too, as does blocking the bot.

Telegram hands each update to whichever caller confirms it first, so only one deployment may read a bot's updates:
live.py's --follow-telegram (bundle variable follow_telegram) turns this off in the others. Nothing here can fail the
scoring: live.main logs any error and carries on.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import broadcast as B
from . import contract as C

log = logging.getLogger(__name__)

FOLLOWERS = "telegram_followers"
FOREVER = 0x7FFFFFFF  # the live_period of a live location shared "until I turn it off"
# The areas as the app's map draws them (built by src/app/scripts/build_area_map.py; the bundle syncs it)
AREAS = Path(__file__).resolve().parents[2] / "app" / "frontend" / "public" / "areas.geojson"
SHORE_KM = 0.3  # a position this close to an area is in it: a phone's error on a beach or a pier. Johor is farther
MAX_PAGES = 20  # getUpdates pages of 100 one run reads at most
PAUSE_S = 0.05  # between messages: Telegram allows a bot about 30 a second
LOCATE = {"keyboard": [[{"text": "📍 Send my location", "request_location": True}]], "resize_keyboard": True}
SHARE_LIVE = "📎, then Location, then Share My Live Location"

FOLLOWERS_COLUMNS = {
    "chat_id": ("BIGINT", "the person's private chat with the bot, which is their Telegram user id: personal data"),
    "started_at": ("TIMESTAMP", "when they first wrote to the bot (SGT)"),
    "active": ("BOOLEAN", "false once they send /stop or block the bot"),
    "mode": ("STRING", "'live' (their live location) or 'pin' (one place they sent); null while not located"),
    "area": ("STRING", "the forecast area they're in (live) or pinned; null outside Singapore. No coordinates kept"),
    "located_at": ("TIMESTAMP", "when Telegram last gave their location (SGT)"),
    "live_until": ("TIMESTAMP", "when the live location they share ends; null for a pin or 'until I turn it off'"),
    "live_message": ("BIGINT", "the message whose edits carry their live location"),
    "told_area": ("STRING", "the area of the last warning they were sent"),
    "told_tier": ("STRING", "its level: Medium or High (Moderate or Elevated in the messages)"),
    "told_since": ("TIMESTAMP", "its first rating, which tells one warning from the next (SGT)"),
    "told_at": ("TIMESTAMP", "when they were sent it (SGT)"),
    "updated_at": ("TIMESTAMP", "when the row last changed (SGT)"),
    "note": ("STRING", "why the bot stopped following them, or the last error sending to them"),
}
_DOC = (
    "People who follow FloodSense's Telegram bot: the forecast area each is in or pinned, never coordinates, and the "
    "last warning each was told. Personal data: only flood-live reads it (ml/follow.py)"
)

WELCOME = (
    "FloodSense warns you when the area you're in may get a PUB flash-flood alert in the next hour.\n\n"
    f"To be followed as you move, share your live location: tap {SHARE_LIVE}, and choose how long "
    "(\"until I turn it off\" works). To follow one place instead, send a pin, or tap 📍 Send my location.\n\n"
    "I'll message you when a warning starts where you are, steps up or down, or when you move into an area under one, "
    "and stay quiet otherwise. I answer within a few minutes. /status shows what I follow, /stop stops.\n\n"
    "I keep only which of Singapore's 47 forecast areas you're in, never your coordinates. FloodSense is not an "
    f"official warning: follow PUB, {B.PUB_CHANNEL}."
)
HELP = (
    "Share your live location to be warned wherever you are in Singapore, or send a pin to follow one place. "
    "/status shows what I follow, /stop stops."
)
STOPPED = "Stopped: I won't message you again. Send /start to begin again."
ENDED = (
    "Your live location ended, so I've stopped following you. Share it again to start, or send a pin to follow one "
    "place."
)
OUTSIDE_LIVE = "You seem to be outside Singapore. I'll follow you once you're back in one of its 47 forecast areas."
OUTSIDE_PIN = "That pin is outside Singapore. FloodSense covers Singapore's 47 forecast areas: send a pin there."


@dataclass(frozen=True)
class Event:
    """One thing a person did in their chat with the bot, at `at` (naive SGT)."""

    chat: int
    kind: str  # start, stop, status, help, located, ended
    at: pd.Timestamp
    message: int | None = None
    lat: float | None = None
    lon: float | None = None
    live: bool = False
    until: pd.Timestamp | None = None  # when a live location ends; None for a pin or one shared until turned off
    edit: bool = False  # a new position of a live location already shared


def sgt(epoch_s: float) -> pd.Timestamp:
    """Naive SGT from epoch seconds, as the tables are read and written (broadcast._sgt)."""
    return pd.Timestamp(epoch_s, unit="s", tz="UTC").tz_convert("Asia/Singapore").tz_localize(None)


def events(updates: list[dict]) -> list[Event]:
    """What people did, in update order: private chats with a person only, the rest ignored."""
    out = []
    for update in sorted(updates, key=lambda u: u["update_id"]):
        edit = "edited_message" in update
        msg = update.get("edited_message") if edit else update.get("message")
        if not msg or (msg.get("chat") or {}).get("type") != "private" or (msg.get("from") or {}).get("is_bot"):
            continue
        chat, mid = int(msg["chat"]["id"]), msg.get("message_id")
        at = sgt(msg.get("edit_date") or msg["date"]) if edit else sgt(msg["date"])
        place = msg.get("location")
        if place:
            period = place.get("live_period")
            if period:
                until = None if period >= FOREVER else sgt(msg["date"] + period)
                out.append(Event(chat, "located", at, mid, place["latitude"], place["longitude"], True, until, edit))
            elif edit:  # a live location no longer active
                out.append(Event(chat, "ended", at, mid))
            else:
                out.append(Event(chat, "located", at, mid, place["latitude"], place["longitude"]))
        elif not edit:
            text = (msg.get("text") or "").strip()
            command = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
            out.append(Event(chat, {"/start": "start", "/stop": "stop", "/status": "status"}.get(command, "help"), at))
    return out


def shapes(path: Path | str = AREAS) -> list[tuple[str, list[list[np.ndarray]]]]:
    """Each area's polygons, each a list of rings of (lon, lat) points, from the app's areas.geojson."""
    out = []
    for feature in json.loads(Path(path).read_text(encoding="utf-8"))["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        out.append((feature["properties"]["area"], [[np.asarray(r, float) for r in p] for p in polygons]))
    return out


def area_at(lat: float, lon: float, areas: list[tuple[str, list[list[np.ndarray]]]]) -> str | None:
    """The forecast area a position is in, or within SHORE_KM of; None outside Singapore."""
    for name, polygons in areas:
        if any(_inside(lon, lat, rings) for rings in polygons):
            return name
    near = [(_km_to(lon, lat, ring), name) for name, polygons in areas for rings in polygons for ring in rings]
    km, name = min(near, default=(np.inf, None))
    return name if km <= SHORE_KM else None


def _inside(x: float, y: float, rings: list[np.ndarray]) -> bool:
    """Even-odd ray casting over a polygon's rings, so a hole is outside."""
    crossings = 0
    for r in rings:
        x1, y1, x2, y2 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        straddles = (y1 > y) != (y2 > y)
        at_x = x1 + (y - y1) * (x2 - x1) / np.where(y2 == y1, 1.0, y2 - y1)
        crossings += int(np.count_nonzero(straddles & (x < at_x)))
    return crossings % 2 == 1


def _km_to(x: float, y: float, ring: np.ndarray) -> float:
    """The distance from a position to a ring's nearest edge, on a flat map around it (Singapore is small)."""
    kx, ky = 111.32 * np.cos(np.radians(y)), 110.57
    ax, ay = (ring[:-1, 0] - x) * kx, (ring[:-1, 1] - y) * ky
    dx, dy = (ring[1:, 0] - ring[:-1, 0]) * kx, (ring[1:, 1] - ring[:-1, 1]) * ky
    t = np.clip(-(ax * dx + ay * dy) / np.maximum(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
    return float(np.min(np.hypot(ax + t * dx, ay + t * dy)))


def blank(chat: int, now: pd.Timestamp) -> dict:
    """A follower who has just written to the bot, not located yet."""
    return {c: None for c in FOLLOWERS_COLUMNS} | {"chat_id": chat, "started_at": now, "active": True}


_UNFOLLOWED = {"mode": None, "area": None, "live_until": None, "live_message": None}


def apply(
    followers: dict[int, dict], evs: list[Event], areas: Callable[[], list], now: pd.Timestamp
) -> tuple[dict[int, dict], dict[int, list], set[int]]:
    """The followers after `evs` and after live locations that have run out: (rows by chat, what to say back by chat,
    in order, as text or (text, keyboard), and the chats that asked for /status). The warnings are `tell`'s."""
    rows = {chat: dict(row) for chat, row in followers.items()}
    say: dict[int, list] = {}
    asked: set[int] = set()
    loaded: list[list] = []

    def area(ev: Event) -> str | None:
        if not loaded:  # read the shapes only for a run that has a position to place
            loaded.append(areas())
        return area_at(ev.lat, ev.lon, loaded[0])

    def change(chat: int, **values) -> None:
        rows[chat] = rows[chat] | values | {"updated_at": now}

    for ev in evs:
        row = rows.get(ev.chat)
        active = bool(row and row["active"])
        if ev.kind == "stop":
            if active:
                change(ev.chat, active=False, note="stopped with /stop", **_UNFOLLOWED)
            say.setdefault(ev.chat, []).append(STOPPED)
            continue
        if ev.edit or ev.kind == "ended":  # only the live location being followed, of someone still following
            if not active or row["mode"] != "live" or row["live_message"] != ev.message:
                continue
            if ev.kind == "ended":
                change(ev.chat, note="live location ended", **_UNFOLLOWED)
                say.setdefault(ev.chat, []).append(ENDED)
            else:
                change(ev.chat, area=area(ev), located_at=ev.at, live_until=ev.until)
            continue
        if ev.kind == "status":
            asked.add(ev.chat)
            continue
        if ev.kind == "help":
            say.setdefault(ev.chat, []).append((HELP, LOCATE))
            continue
        if not active:  # /start or a location starts following, again after a /stop
            started = (row or {}).get("started_at") or now
            rows[ev.chat] = blank(ev.chat, now) | {"started_at": started, "updated_at": now}
        if ev.kind == "start":
            say.setdefault(ev.chat, []).append((WELCOME, LOCATE))
            continue
        here = area(ev)
        if ev.live:
            change(ev.chat, mode="live", area=here, located_at=ev.at, live_until=ev.until, live_message=ev.message)
            say.setdefault(ev.chat, []).append(following(rows[ev.chat]) if here else OUTSIDE_LIVE)
        elif here:
            change(ev.chat, mode="pin", area=here, located_at=ev.at, live_until=None, live_message=None)
            say.setdefault(ev.chat, []).append(following(rows[ev.chat]))
        else:
            say.setdefault(ev.chat, []).append(OUTSIDE_PIN)
    for chat, row in rows.items():
        until = row["live_until"]
        if row["active"] and row["mode"] == "live" and until is not None and not pd.isna(until) and until <= now:
            change(chat, note="live location ended", **_UNFOLLOWED)
            say.setdefault(chat, []).append(ENDED)
    return rows, say, asked


def following(row: dict) -> str:
    """The answer to a location just shared."""
    if row["mode"] == "pin":
        return (
            f"Following {row['area']}, where your pin is. I'll message you when a warning starts there. To be "
            f"followed as you move instead, share your live location: {SHARE_LIVE}."
        )
    until = row["live_until"]
    ends = f", until {B.clock(until)} when your live location ends" if until is not None and not pd.isna(until) else ""
    return (
        f"Following you{ends}. You're in {row['area']} now. I'll message you when a warning starts there or wherever "
        "you move. Stop sharing your location, or send /stop, to end it."
    )


def tell(rows: dict[int, dict], warnings: pd.DataFrame, tick: pd.Timestamp) -> dict[int, tuple[str, tuple]]:
    """For each located follower whose area is warned: the message, and the warning it tells (area, tier, since), if
    they haven't been told that warning at that level in that area."""
    by_area = {w.area: w for w in warnings.itertuples(index=False)}
    out = {}
    for chat, row in rows.items():
        w = by_area.get(row["area"]) if row["active"] and row["area"] else None
        if w is None:
            continue
        same = row["told_area"] == w.area and _same_time(row["told_since"], w.since)
        if same and row["told_tier"] == w.tier:
            continue
        out[chat] = (warning_text(row, w, row["told_tier"] if same else None, tick), (w.area, w.tier, w.since))
    return out


def warning_text(row: dict, w, previous: str | None, tick: pd.Timestamp) -> str:
    """One follower's warning, in the app's words (broadcast.SHOWN, MEANING, ADVICE)."""
    where = f"where you are, in {w.area}," if row["mode"] == "live" else f"in {w.area}, where your pin is,"
    step = f" ({'down' if B.RANK[previous] > B.RANK[w.tier] else 'up'} from {B.SHOWN[previous]})" if previous else ""
    lines = [
        f"{B.ICON[w.tier]} {B.SHOWN[w.tier]} {where} until {B.clock(w.until)}{step}",
        B.MEANING[w.tier],
        "",
        f"What to do: {B.ADVICE[w.tier]}",
    ]
    if isinstance(w.reason, str) and w.reason:
        lines.append(f"Why: {w.reason}")
    lines += ["", f"FloodSense rated it at {B.clock(tick)}. Not an official warning: follow PUB, {B.PUB_CHANNEL}."]
    return "\n".join(lines)


def status(row: dict | None, warnings: pd.DataFrame | None) -> str:
    """The answer to /status."""
    if not row or not row["active"]:
        return "I'm not following you. Share your live location, or send a pin, to start."
    if row["mode"] is None:
        return f"You haven't shared a location yet. Share your live location ({SHARE_LIVE}), or send a pin."
    if row["area"] is None:
        return OUTSIDE_LIVE
    what = (
        f"Following you by your live location, last updated at {B.clock(row['located_at'])}: you're in {row['area']}."
        if row["mode"] == "live"
        else f"Following {row['area']}, where your pin is."
    )
    if warnings is None:
        return f"{what} FloodSense has no fresh rating right now."
    w = next((w for w in warnings.itertuples(index=False) if w.area == row["area"]), None)
    return f"{what} " + (
        f"{B.ICON[w.tier]} {B.SHOWN[w.tier]} there until {B.clock(w.until)}." if w else "No warning there now."
    )


def handle(
    bot,
    followers: dict[int, dict],
    areas: Callable[[], list],
    warnings: pd.DataFrame | None,
    tick: pd.Timestamp | None,
    now: pd.Timestamp,
) -> tuple[dict[int, dict], dict, int | None]:
    """One run of the bot, without Spark: read the updates, answer them, and tell followers their warnings (`warnings`
    in force at `tick`, None when the tick isn't fresh). Returns the rows that changed, a summary for the run's log,
    and the offset that confirms the updates read (None when there were none)."""
    updates, offset = [], None
    for _ in range(MAX_PAGES):
        page = bot.updates(offset=offset)
        updates += page
        if page:
            offset = page[-1]["update_id"] + 1
        if len(page) < 100:
            break
    rows, say, asked = apply(followers, events(updates), areas, now)
    told = tell(rows, warnings, tick) if warnings is not None else {}
    sent = failed = 0
    for chat in sorted(set(say) | set(told) | asked):
        messages = list(say.get(chat, []))
        if chat in told:
            messages.append(told[chat][0])
        if chat in asked:
            messages.append(status(rows.get(chat), warnings))
        for message in messages:
            text, keyboard = message if isinstance(message, tuple) else (message, None)
            try:
                bot.message(chat, text, keyboard)
                sent += 1
            except Exception as exc:  # noqa: BLE001 - one chat mustn't stop the rest
                failed += 1
                reason = str(exc)[:200]
                log.warning("a message to a follower wasn't sent: %s", reason)
                if chat in rows and reason.startswith("HTTP 403"):  # blocked the bot, or deleted their account
                    rows[chat] = rows[chat] | {"active": False, "note": reason, "updated_at": now} | _UNFOLLOWED
                break
            time.sleep(PAUSE_S)
        else:
            if chat in told:
                area, tier, since = told[chat][1]
                told_now = {"told_area": area, "told_tier": tier, "told_since": since, "told_at": now}
                rows[chat] = rows[chat] | told_now | {"updated_at": now}
    changed = {chat: row for chat, row in rows.items() if row != followers.get(chat)}
    summary = {
        "updates": len(updates),
        "following": sum(bool(r["active"] and r["area"]) for r in rows.values()),
        "live": sum(bool(r["active"] and r["mode"] == "live") for r in rows.values()),
        "warned": len(told),
        "messages": sent,
        "failed": failed,
    }
    return changed, summary, offset


# --- on Databricks ---------------------------------------------------------------------------------------------------


def run(spark, *, app: str, ml: str, model: str, tick, now, bot: B.Telegram, areas: Path | str = AREAS) -> dict:
    """The bot's run after broadcast.run: answer what people sent since the last run, tell followers their warnings at
    `tick` (the newest tick `model` scored) when it is fresh, and keep the followers in <app>.telegram_followers.
    `areas` is the app's areas.geojson, which places a position in an area."""
    table = f"{app}.{FOLLOWERS}"
    if not B._ensure(spark, table, FOLLOWERS_COLUMNS, _DOC):
        return {"skipped": f"no schema for {table}"}
    fresh = tick is not None and not pd.isna(tick) and now - pd.Timestamp(tick) <= B.FRESH
    warnings = B.in_force(B._ratings(spark, f"{ml}.{C.PREDICTIONS}", model, tick), tick) if fresh else None
    changed, summary, offset = handle(
        bot,
        _followers(spark, table),
        lambda: shapes(areas),
        warnings,
        tick,
        now,
    )
    if changed:
        _merge(spark, table, list(changed.values()))
    if offset is not None:
        bot.updates(offset=offset, limit=1)  # confirms every update this run read
    return summary if fresh else summary | {"warnings": "no fresh tick"}


def _followers(spark, table: str) -> dict[int, dict]:
    stamps = [c for c, (kind, _) in FOLLOWERS_COLUMNS.items() if kind == "TIMESTAMP"]
    columns = ", ".join(f"unix_timestamp({c}) AS {c}" if c in stamps else c for c in FOLLOWERS_COLUMNS)
    out = {}
    for r in spark.sql(f"SELECT {columns} FROM {table}").collect():
        row = r.asDict()
        out[int(row["chat_id"])] = row | {c: None if row[c] is None else sgt(row[c]) for c in stamps}
    return out


def _merge(spark, table: str, rows: list[dict]) -> None:
    """Upsert the rows that changed, by chat."""
    schema = ", ".join(f"{c} {kind}" for c, (kind, _) in FOLLOWERS_COLUMNS.items())
    values = [tuple(_value(row[c]) for c in FOLLOWERS_COLUMNS) for row in rows]
    spark.createDataFrame(values, schema=schema).createOrReplaceTempView("telegram_followers_changed")
    spark.sql(
        f"""MERGE INTO {table} t USING telegram_followers_changed s ON t.chat_id = s.chat_id
            WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"""
    )


def _value(v):
    """A Python value Spark takes: None for NaN and NaT, plain ints, and times made SGT-aware. Spark would read a naive
    datetime in the process's own time zone (UTC on serverless), not the session's."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return (v if v.tzinfo else v.tz_localize("Asia/Singapore")).to_pydatetime()
    if isinstance(v, np.integer | np.bool_):
        return v.item()
    return v


def _same_time(a, b) -> bool:
    return a is not None and b is not None and not pd.isna(a) and pd.Timestamp(a) == pd.Timestamp(b)
