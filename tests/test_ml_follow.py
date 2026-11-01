"""follow: the Telegram bot's followers, from the updates Telegram gives to the warnings they hear, checked without
Spark or the network."""

import io
import json
import urllib.error

import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import broadcast as B
from daisi_team_master_corgis_floodsense.ml import follow as F

DAY = "2026-10-31"
CHAT, OTHER = 1001, 1002
SHAPES = F.shapes()  # the app's areas.geojson
NEAR = {"Bedok": (1.3235, 103.9301), "Geylang": (1.3170, 103.8870), "Tampines": (1.3501, 103.9441)}
# Johor Bahru, 4.9 km from Woodlands' label point: nearer than much of Singapore is to its own
JOHOR = (1.4655, 103.7578)


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(F, "PAUSE_S", 0)


def at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY} {hhmm}")


def epoch(hhmm: str) -> int:
    return int(at(hhmm).tz_localize("Asia/Singapore").timestamp())


class Updates:
    """Telegram's updates as getUpdates returns them, numbered in order."""

    def __init__(self):
        self.items = []

    def _add(self, kind: str, msg: dict, chat: int, hhmm: str, edited: str | None = None):
        msg = {"message_id": msg.pop("message_id", len(self.items) + 1), "date": epoch(hhmm), **msg}
        msg |= {"chat": {"id": chat, "type": "private"}, "from": {"id": chat, "is_bot": False}}
        if edited:
            msg["edit_date"] = epoch(edited)
        self.items.append({"update_id": 500 + len(self.items), kind: msg})
        return self

    def text(self, chat, text, hhmm="10:00"):
        return self._add("message", {"text": text}, chat, hhmm)

    def live(self, chat, place, hhmm="10:00", period=8 * 3600, message_id=77):
        lat, lon = NEAR.get(place, place) if isinstance(place, str) else place
        location = {"latitude": lat, "longitude": lon, "live_period": period}
        return self._add("message", {"message_id": message_id, "location": location}, chat, hhmm)

    def moved(self, chat, place, hhmm, shared="10:00", message_id=77, period=8 * 3600):
        lat, lon = NEAR[place]
        location = {"latitude": lat, "longitude": lon, "live_period": period}
        return self._add("edited_message", {"message_id": message_id, "location": location}, chat, shared, hhmm)

    def stopped_sharing(self, chat, hhmm, shared="10:00", message_id=77):
        lat, lon = NEAR["Bedok"]
        location = {"latitude": lat, "longitude": lon}
        return self._add("edited_message", {"message_id": message_id, "location": location}, chat, shared, hhmm)

    def pin(self, chat, place, hhmm="10:00"):
        lat, lon = NEAR.get(place, place) if isinstance(place, str) else place
        return self._add("message", {"location": {"latitude": lat, "longitude": lon}}, chat, hhmm)


def warned(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """Warnings in force: (area, tier, since, issued)."""
    return pd.DataFrame(
        [
            {"area": a, "tier": t, "since": at(s), "issued": at(i), "until": at(i) + B.WARNING, "p_alert": 0.06,
             "reason": "heavy rain at nearby gauges"}
            for a, t, s, i in rows
        ]
    )  # fmt: skip


class FakeBot:
    def __init__(self, updates=(), blocked=()):
        self.pending, self.blocked = list(updates), set(blocked)
        self.sent, self.asked = [], []

    def updates(self, offset=None, limit=100):
        self.asked.append(offset)
        return [u for u in self.pending if offset is None or u["update_id"] >= offset][:limit]

    def message(self, chat, text, keyboard=None):
        if chat in self.blocked:
            raise RuntimeError("HTTP 403: Forbidden: bot was blocked by the user")
        self.sent.append((chat, text, keyboard))


def run(followers, updates, warnings=None, now="10:30", tick="10:30", bot=None):
    bot = bot or FakeBot(updates.items if isinstance(updates, Updates) else updates)
    changed, summary, offset = F.handle(bot, followers, lambda: SHAPES, warnings, at(tick), at(now))
    return {**followers, **changed}, bot, summary, offset


def texts(bot, chat=CHAT) -> list[str]:
    return [text for c, text, _ in bot.sent if c == chat]


# --- what people did --------------------------------------------------------------------------------------------------


def test_commands_locations_and_their_edits_become_events_in_update_order():
    u = Updates().text(CHAT, "/start follow").live(CHAT, "Bedok", period=F.FOREVER).moved(CHAT, "Geylang", "10:20")
    u.stopped_sharing(CHAT, "11:00").pin(CHAT, "Tampines", "11:05").text(CHAT, "/STOP@daisi_floodsense_bot")
    u.text(CHAT, "thanks!")
    evs = F.events(list(reversed(u.items)))
    assert [e.kind for e in evs] == ["start", "located", "located", "ended", "located", "stop", "help"]
    share, moved, ended, pin = evs[1:5]
    assert share.live and share.until is None and not share.edit  # shared until turned off
    assert moved.live and moved.edit and moved.at == at("10:20") and moved.message == share.message
    assert ended.message == share.message and ended.at == at("11:00")
    assert not pin.live and pin.until is None


def test_a_live_location_for_a_while_ends_that_long_after_it_was_shared():
    (e,) = F.events(Updates().live(CHAT, "Bedok", "10:00", period=3600).items)
    assert e.until == at("11:00")


def test_groups_channels_and_bots_are_ignored():
    u = Updates().text(CHAT, "/start")
    u.items[0]["message"]["chat"]["type"] = "group"
    u.text(OTHER, "/start")
    u.items[1]["message"]["from"]["is_bot"] = True
    u.items.append({"update_id": 999, "channel_post": {"text": "hi"}})
    assert F.events(u.items) == []


def test_a_position_is_in_the_area_the_map_draws_it_in_and_in_none_outside_singapore():
    assert [F.area_at(*NEAR[a], SHAPES) for a in NEAR] == list(NEAR)
    assert F.area_at(1.2494, 103.8303, SHAPES) == "Sentosa"
    assert F.area_at(*JOHOR, SHAPES) is None and F.area_at(1.4630, 103.7650, SHAPES) is None  # JB and its checkpoint
    assert F.area_at(1.3030, 103.93, SHAPES) == "Bedok"  # 150 m off East Coast Park: a phone's error on the beach
    assert F.area_at(1.2985, 103.93, SHAPES) is None  # 600 m out at sea


# --- following --------------------------------------------------------------------------------------------------------


def test_start_welcomes_with_a_location_button_and_a_live_location_is_followed():
    followers, bot, summary, offset = run({}, Updates().text(CHAT, "/start").live(CHAT, "Bedok", "10:01"))
    row = followers[CHAT]
    assert (row["active"], row["mode"], row["area"], row["live_message"]) == (True, "live", "Bedok", 77)
    assert row["live_until"] == at("18:01") and row["located_at"] == at("10:01")
    welcome, following = bot.sent
    assert welcome[1] == F.WELCOME and welcome[2] == F.LOCATE
    assert following[1].startswith("Following you, until 6:01 pm when your live location ends. You're in Bedok now.")
    assert offset == 502 and summary["updates"] == 2 and summary["live"] == 1
    assert "lat" not in json.dumps(row, default=str)  # no coordinates kept


def test_new_positions_move_the_area_silently_and_an_older_share_is_ignored():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok"))
    u = Updates().moved(CHAT, "Geylang", "10:10").moved(CHAT, "Tampines", "10:20", message_id=12)
    followers, bot, *_ = run(followers, u)
    assert followers[CHAT]["area"] == "Geylang" and followers[CHAT]["located_at"] == at("10:10")
    assert bot.sent == []


def test_sharing_that_stops_or_runs_out_ends_the_following_with_one_message():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok", period=3600))
    stopped, bot, *_ = run(followers, Updates().stopped_sharing(CHAT, "10:40"), now="10:45")
    assert (stopped[CHAT]["mode"], stopped[CHAT]["area"], stopped[CHAT]["active"]) == (None, None, True)
    assert texts(bot) == [F.ENDED]
    lapsed, bot, *_ = run(followers, [], now="11:00")
    assert lapsed[CHAT]["area"] is None and texts(bot) == [F.ENDED]
    _, bot, *_ = run(lapsed, [], now="11:05")
    assert bot.sent == []  # said once


def test_a_pin_follows_one_place_and_a_pin_outside_singapore_follows_nothing():
    followers, bot, *_ = run({}, Updates().pin(CHAT, "Tampines").pin(OTHER, JOHOR))
    assert (followers[CHAT]["mode"], followers[CHAT]["area"]) == ("pin", "Tampines")
    assert texts(bot)[0].startswith("Following Tampines, where your pin is.")
    assert followers[OTHER]["area"] is None and texts(bot, OTHER) == [F.OUTSIDE_PIN]


def test_a_live_location_outside_singapore_is_followed_from_when_it_comes_back():
    followers, bot, *_ = run({}, Updates().live(CHAT, JOHOR))
    assert followers[CHAT]["mode"] == "live" and followers[CHAT]["area"] is None and texts(bot) == [F.OUTSIDE_LIVE]
    followers, bot, *_ = run(followers, Updates().moved(CHAT, "Geylang", "10:20"))
    assert followers[CHAT]["area"] == "Geylang" and bot.sent == []


def test_stop_stops_everything_until_start_and_status_alone_follows_no_one():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok"))
    started = followers[CHAT]["started_at"]
    followers, bot, *_ = run(followers, Updates().text(CHAT, "/stop").moved(CHAT, "Geylang", "10:20"))
    assert followers[CHAT]["active"] is False and followers[CHAT]["area"] is None and texts(bot) == [F.STOPPED]
    followers, bot, *_ = run(followers, Updates().text(CHAT, "/start"), now="12:00")
    assert followers[CHAT]["active"] and followers[CHAT]["started_at"] == started
    followers, bot, *_ = run(followers, Updates().text(OTHER, "/status"))
    assert OTHER not in followers and texts(bot, OTHER) == [F.status(None, None)]


# --- warnings where they are ------------------------------------------------------------------------------------------


def test_a_follower_hears_a_warning_where_they_are_once_and_again_when_it_steps_up():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok"), now="10:00")
    moderate = warned(("Bedok", "Medium", "10:25", "10:25"))
    followers, bot, summary, _ = run(followers, [], moderate, now="10:30")
    (text,) = texts(bot)
    assert text.startswith("🟠 Moderate where you are, in Bedok, until 11:25 am\n")
    assert B.MEANING["Medium"] in text and f"What to do: {B.ADVICE['Medium']}" in text
    assert "Why: heavy rain at nearby gauges" in text and summary["warned"] == 1
    assert (followers[CHAT]["told_area"], followers[CHAT]["told_tier"]) == ("Bedok", "Medium")
    followers, bot, *_ = run(followers, [], moderate, now="10:35")
    assert bot.sent == []  # holding at one level: quiet
    elevated = warned(("Bedok", "High", "10:25", "10:40"))
    followers, bot, *_ = run(followers, [], elevated, now="10:40")
    assert texts(bot)[0].startswith("🔴 Elevated where you are, in Bedok, until 11:40 am (up from Moderate)")


def test_moving_into_a_warned_area_is_news_and_a_new_warning_after_one_lapsed_too():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok"), now="10:00")
    both = warned(("Bedok", "Medium", "10:25", "10:25"), ("Geylang", "High", "10:20", "10:30"))
    followers, bot, *_ = run(followers, [], both, now="10:30")
    followers, bot, *_ = run(followers, Updates().moved(CHAT, "Geylang", "10:33"), both, now="10:35")
    assert texts(bot)[0].startswith("🔴 Elevated where you are, in Geylang, until 11:30 am\n")
    later = warned(("Geylang", "Medium", "13:00", "13:00"))
    followers, bot, *_ = run(followers, [], later, now="13:00", tick="13:00")
    assert texts(bot)[0].startswith("🟠 Moderate where you are, in Geylang")


def test_a_pinned_place_is_named_as_the_pin_and_no_fresh_tick_tells_nothing():
    followers, *_ = run({}, Updates().pin(CHAT, "Tampines"), now="10:00")
    w = warned(("Tampines", "High", "10:25", "10:25"))
    followers, bot, *_ = run(followers, [], None, now="10:30")
    assert bot.sent == []
    followers, bot, *_ = run(followers, [], w, now="10:30")
    assert texts(bot)[0].startswith("🔴 Elevated in Tampines, where your pin is, until 11:25 am")


def test_status_says_what_is_followed_and_the_warning_there():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok", "10:01"), now="10:02")
    w = warned(("Bedok", "High", "10:25", "10:25"))
    followers, bot, *_ = run(followers, Updates().text(CHAT, "/status"), w, now="10:30")
    warning, status = texts(bot)
    assert warning.startswith("🔴 Elevated where you are")
    assert status == (
        "Following you by your live location, last updated at 10:01 am: you're in Bedok. 🔴 Elevated there until "
        "11:25 am."
    )
    assert F.status(followers[CHAT], None).endswith("FloodSense has no fresh rating right now.")


def test_a_follower_who_blocked_the_bot_is_stopped_and_a_failed_warning_is_tried_again():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok").live(OTHER, "Geylang"), now="10:00")
    w = warned(("Bedok", "Medium", "10:25", "10:25"), ("Geylang", "Medium", "10:25", "10:25"))
    followers, _, summary, _ = run(followers, [], w, now="10:30", bot=FakeBot(blocked={CHAT}))
    assert followers[CHAT]["active"] is False and followers[CHAT]["note"].startswith("HTTP 403")
    assert followers[CHAT]["told_tier"] is None and summary["failed"] == 1
    assert followers[OTHER]["told_tier"] == "Medium" and summary["messages"] == 1


def test_the_updates_are_read_page_by_page_and_the_last_one_sets_the_offset():
    u = Updates()
    for i in range(130):
        u.moved(CHAT, "Geylang" if i % 2 else "Bedok", "10:20")
    bot = FakeBot(u.items)
    *_, summary, offset = run({}, [], bot=bot)
    assert bot.asked == [None, 600] and summary["updates"] == 130 and offset == 630


def test_unchanged_followers_are_not_written_again():
    followers, *_ = run({}, Updates().live(CHAT, "Bedok"), now="10:00")
    changed, *_ = F.handle(FakeBot(), followers, lambda: SHAPES, None, at("10:30"), at("10:30"))
    assert changed == {}


# --- the bot's calls --------------------------------------------------------------------------------------------------


def capture(monkeypatch, result=None):
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        calls.append((request.full_url, dict(pair.split("=", 1) for pair in request.data.decode().split("&"))))
        return Resp(json.dumps({"ok": True, "result": result or []}).encode())

    monkeypatch.setattr(B.urllib.request, "urlopen", urlopen)
    return calls


def test_the_bot_asks_for_private_messages_and_their_edits_and_sends_a_keyboard(monkeypatch):
    calls = capture(monkeypatch, result=[{"update_id": 7}])
    bot = B.Telegram("123:SECRET-TOKEN")
    assert bot.updates(offset=8, limit=1) == [{"update_id": 7}]
    bot.message(CHAT, "hello", F.LOCATE)
    (url, body), (_, sent) = calls
    assert url.endswith("/getUpdates") and body["offset"] == "8" and body["limit"] == "1"
    assert json.loads(B.urllib.parse.unquote_plus(body["allowed_updates"])) == ["message", "edited_message"]
    assert json.loads(B.urllib.parse.unquote_plus(sent["reply_markup"])) == F.LOCATE and sent["chat_id"] == str(CHAT)


def test_a_bot_error_gives_its_reason_and_never_the_token(monkeypatch):
    def refuse(request, timeout):
        body = io.BytesIO(json.dumps({"ok": False, "description": "Forbidden: bot was blocked by the user"}).encode())
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, body)

    monkeypatch.setattr(B.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError) as err:
        B.Telegram("123:SECRET-TOKEN").message(CHAT, "hi")
    assert str(err.value) == "HTTP 403: Forbidden: bot was blocked by the user" and "SECRET" not in repr(err.value)
