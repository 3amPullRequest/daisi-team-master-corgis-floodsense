"""What FloodSense sends out when it warns: a post to its Telegram channel, and an email or a text to everyone who asked
the app for warnings in that area.

Every Moderate or Elevated rating is a warning for the next hour, as the model was trained (a PUB alert within
(t, t + 60 min]). So an area's warning stays in force until an hour after the last rating that issued it, and ratings
at most an hour apart are one warning. Elevated has priority: while any Elevated rating is under an hour old the area
is at Elevated, so a Moderate warning can follow an Elevated one only an hour after the last Elevated rating.
`in_force` gives each area's warning at a tick. The app's live map (backend /api/now), its storm replays (frontend
lib/warnings.ts), the live_risk view (publish.py) and the scoreboard's warning episodes (scoreboard.py) count
warnings the same way.

flood-live's score task calls `run` after each run. `due` decides what to send: a post when an area's warning starts,
steps up to Elevated or steps down to Moderate, and nothing while it holds at one level. One Telegram message per run
lists every area that changed. Each subscriber gets one email or text per run for the areas they chose, and a welcome
message once they sign up. ml.warning_posts logs every post, which also keeps a re-run from posting twice. Nothing is
sent for a tick more than FRESH old: after an outage, old warnings are history, not news.

The app's Now view writes the subscriptions (an email address or a Singapore mobile number, and the areas) to
<app schema>.warning_subscriptions, apart from the analytics schemas because it holds personal data. A sender with no
credentials in the secret scope (telegram-bot-token; parser-alert-smtp-user and -password; twilio-account-sid,
-auth-token and -from-number) is skipped, and the run's summary says so. People who follow the bot in a private chat
hear about the warnings wherever they are from ml/follow.py. Nothing here can fail the scoring: live.main logs any error
and carries on.

Try the senders without a storm, as your profile:

    uv run python -m daisi_team_master_corgis_floodsense.ml.broadcast --profile flood_dev --test-telegram
    uv run python -m daisi_team_master_corgis_floodsense.ml.broadcast --profile flood_dev --test-email you@example.com
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage

import numpy as np
import pandas as pd

from ..ingest.cli import read_secret
from ..ingest.notify import SMTP_PASSWORD_SECRET, SMTP_USER_SECRET, Mailer
from . import contract as C

log = logging.getLogger(__name__)

WARNING = pd.Timedelta(minutes=C.HORIZON_MINUTES)  # how long one rating's warning lasts
HISTORY = pd.Timedelta(hours=6)  # how far back a warning's first rating is looked for
FRESH = pd.Timedelta(minutes=20)  # the app calls a forecast this old paused (NowView STALE_AFTER_MIN)
WELCOME_FOR = pd.Timedelta(days=1)  # a welcome that can't be sent is retried this long
POSTS = "warning_posts"
SUBSCRIPTIONS = "warning_subscriptions"
CHANNEL = "@daisi_floodsense"
TELEGRAM_SECRET = "telegram-bot-token"
TWILIO_SECRETS = ("twilio-account-sid", "twilio-auth-token", "twilio-from-number")
PUB_CHANNEL = "t.me/pubfloodalerts"
RANK = {"Low": 0, "Medium": 1, "High": 2}
# the app's words (frontend lib/content.ts: TIER_LABEL, TIER_MEANING, and ADVICE for people out on foot)
SHOWN = {"Medium": "Moderate", "High": "Elevated"}
ICON = {"Medium": "🟠", "High": "🔴"}
MEANING = {
    "Medium": "A PUB flash-flood alert is possible there in the next hour.",
    "High": "A PUB flash-flood alert is more likely there in the next hour.",
}
ADVICE = {
    "Medium": "No need to change plans yet. If the rain gets heavy, keep away from underpasses and paths beside canals.",
    "High": "Allow extra time. Keep to sheltered walkways, and avoid underpasses and paths beside drains and canals.",
}
STOP = "To stop these messages, open FloodSense, go to Now, and choose Stop under Warnings by email or text."
_ID = re.compile(r"^[0-9a-f-]{36}$")

POSTS_COLUMNS = {
    "area": ("STRING", "NEA 2-hour forecast area (one of 47)"),
    "tick": ("TIMESTAMP", "the 5-minute tick the post went out for (SGT)"),
    "tier": ("STRING", "Medium or High: the warning posted (the app shows Moderate or Elevated)"),
    "previous": ("STRING", "the level last posted in the same warning, when it stepped up or down; null for a new one"),
    "since": ("TIMESTAMP", "the warning's first rating (SGT)"),
    "until": ("TIMESTAMP", "when the level lapses unless FloodSense rates the area again: an hour after its rating"),
    "model": ("STRING", "the FloodSense version whose ratings issued it (ml.predictions.model)"),
    "posted_at": ("TIMESTAMP", "when the run sent it (SGT)"),
    "telegram": ("STRING", "'sent', or why the channel post didn't go out"),
    "emails": ("BIGINT", "subscribers emailed about this area"),
    "texts": ("BIGINT", "subscribers texted about this area"),
}
SUBSCRIPTIONS_COLUMNS = {
    "id": ("STRING", "the subscription's random id, which the app shows its owner"),
    "created_at": ("TIMESTAMP", "when it was made"),
    "created_by": ("STRING", "the Databricks user signed in to the app who made it, if any"),
    "channel": ("STRING", "email or sms"),
    "address": ("STRING", "the email address or Singapore mobile number (+65...) to send to: personal data"),
    "areas": ("ARRAY<STRING>", "the NEA forecast areas to send warnings for"),
    "places": ("ARRAY<STRING>", "the places the person chose those areas by"),
    "active": ("BOOLEAN", "false once its owner stops it"),
    "stopped_at": ("TIMESTAMP", "when its owner stopped it"),
    "welcome": ("STRING", "'sent', or why the welcome message hasn't gone out yet"),
    "welcomed_at": ("TIMESTAMP", "when the welcome message went out"),
}


# --- which warnings are in force, and which are news ---------------------------------------------------------------


def in_force(ratings: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """Each area's warning in force at tick `at`, from one model's ratings (area, ts, tier, p_alert, reason; Low rows
    may be left out): tier (Medium, or High while an Elevated rating is under an hour old), since (the warning's first
    rating), issued (the rating that holds the tier: the newest Elevated one at High), until (an hour after it), and
    that rating's p_alert and reason. An area with no warning in force has no row."""
    columns = ["area", "tier", "since", "issued", "until", "p_alert", "reason"]
    at = pd.Timestamp(at)
    warned = ratings[(ratings["tier"] != "Low") & (ratings["ts"] <= at)].sort_values("ts")
    rows = []
    for area, part in warned.groupby("area", sort=True):
        live = part[part["ts"] > at - WARNING]
        if live.empty:
            continue
        high = live[live["tier"] == "High"]
        issued = (high if len(high) else live).iloc[-1]
        ts = part["ts"].to_numpy()
        first = len(ts) - 1  # the newest warned rating, in force at `at`: back through ratings at most an hour apart
        while first > 0 and ts[first] - ts[first - 1] <= WARNING.to_timedelta64():
            first -= 1
        rows.append(
            {
                "area": area,
                "tier": "High" if len(high) else "Medium",
                "since": pd.Timestamp(ts[first]),
                "issued": issued["ts"],
                "until": issued["ts"] + WARNING,
                "p_alert": issued.get("p_alert"),
                "reason": issued.get("reason"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def due(warnings: pd.DataFrame, posted: pd.DataFrame) -> pd.DataFrame:
    """The warnings to send now, with `previous`: an area with no post since its warning began gets one (previous
    None), and so does an area whose last post was at the other level (previous that level). `posted` holds each
    area's last post (area, tick, tier)."""
    last = {row.area: row for row in posted.itertuples(index=False)}
    keep, previous = [], []
    for w in warnings.itertuples(index=False):
        p = last.get(w.area)
        new = p is None or pd.Timestamp(p.tick) < w.since
        keep.append(new or p.tier != w.tier)
        previous.append(None if new else p.tier)
    out = warnings.assign(previous=previous)
    return out[np.array(keep, dtype=bool)].reset_index(drop=True)


# --- the messages ----------------------------------------------------------------------------------------------------


def clock(ts) -> str:
    """'2:45 pm', as the app writes times."""
    ts = pd.Timestamp(ts)
    return f"{ts.hour % 12 or 12}:{ts.minute:02d} {'am' if ts.hour < 12 else 'pm'}"


def joined(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def lines(sent: pd.DataFrame) -> list[str]:
    """Elevated first; areas with the same level, end and step share a line."""
    out = []
    for tier in ("High", "Medium"):
        part = sent[sent["tier"] == tier]
        keys = part.assign(prev=part["previous"].fillna("")).groupby(["until", "prev"], sort=True)
        for (until, prev), group in keys:
            step = f" ({'down' if RANK[prev] > RANK[tier] else 'up'} from {SHOWN[prev]})" if prev else ""
            out.append(f"{ICON[tier]} {SHOWN[tier]} in {joined(sorted(group['area']))} until {clock(until)}{step}")
            out.append(MEANING[tier])
    return out


def channel_post(sent: pd.DataFrame, tick: pd.Timestamp) -> str:
    """The Telegram post for one run."""
    body = "\n".join(lines(sent))
    return f"FloodSense at {clock(tick)}\n\n{body}\n\nNot an official warning. Follow PUB: {PUB_CHANNEL}"


def by_level(sent: pd.DataFrame) -> list[str]:
    """'Elevated in Geylang and Kallang', then 'Moderate in Bedok'."""
    return [
        f"{SHOWN[tier]} in {joined(sorted(sent.loc[sent['tier'] == tier, 'area']))}"
        for tier in ("High", "Medium")
        if (sent["tier"] == tier).any()
    ]


def email_warning(address: str, sent: pd.DataFrame, tick: pd.Timestamp) -> EmailMessage:
    areas = sorted(sent["area"])
    advice = [f"What to do at {SHOWN[t]}: {ADVICE[t]}" for t in ("High", "Medium") if (sent["tier"] == t).any()]
    why = [f"Why, in {r.area}: {r.reason}" for r in sent.itertuples() if isinstance(r.reason, str) and r.reason]
    message = EmailMessage()
    message["Subject"] = "FloodSense: " + ", ".join(by_level(sent))
    message["To"] = address
    message.set_content(
        "\n".join(
            [
                *lines(sent),
                "",
                *advice,
                *(["", *why] if why else []),
                "",
                f"FloodSense rated these at {clock(tick)}. It is not an official warning: follow PUB Flood Alerts on "
                f"Telegram, {PUB_CHANNEL}.",
                "",
                f"You get this because you asked FloodSense for warnings for {joined(areas)}. {STOP}",
            ]
        )
    )
    return message


def text_warning(sent: pd.DataFrame, tick: pd.Timestamp) -> str:
    """One text, short enough for a single SMS when one area is warned."""
    until = clock(sent["until"].max())
    return f"FloodSense {clock(tick)}: {'; '.join(by_level(sent))} until {until}. Not official: follow PUB on Telegram."


def welcome_email(address: str, areas: list[str]) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = f"FloodSense warnings for {joined(areas)}"
    message["To"] = address
    message.set_content(
        "\n".join(
            [
                f"You'll get an email when FloodSense issues a Moderate or Elevated warning for {joined(areas)}.",
                "",
                "Each warning covers the next hour. FloodSense rates every area every 5 minutes, and emails again "
                "if a warning steps up or down. It sends nothing while all is quiet.",
                "",
                f"FloodSense is not an official warning: follow PUB Flood Alerts on Telegram, {PUB_CHANNEL}.",
                "",
                STOP,
            ]
        )
    )
    return message


def welcome_text(areas: list[str]) -> str:
    return (
        f"FloodSense: you'll get a text when it warns for {joined(areas)}. Each warning covers the next hour. "
        "Not official: follow PUB on Telegram."
    )


# --- the senders -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Telegram:
    """A bot that is an administrator of the channel, allowed to post (its token from @BotFather). It also talks to
    the people who follow it in private chats (ml/follow.py)."""

    token: str
    chat: str = CHANNEL

    def send(self, text: str) -> None:
        self.message(self.chat, text)

    def message(self, chat: int | str, text: str, keyboard: dict | None = None) -> None:
        body = {"chat_id": chat, "text": text, "disable_web_page_preview": "true"}
        if keyboard:
            body["reply_markup"] = json.dumps(keyboard)
        _post(f"https://api.telegram.org/bot{self.token}/sendMessage", body, None, "description")

    def updates(self, offset: int | None = None, limit: int = 100) -> list[dict]:
        """The private messages people sent the bot, and their edits, that no call has confirmed yet (getUpdates).
        Asking with `offset` confirms every update before it; Telegram keeps the rest for 24 hours."""
        body = {"timeout": 0, "limit": limit, "allowed_updates": json.dumps(["message", "edited_message"])}
        if offset is not None:
            body["offset"] = offset
        found = _post(f"https://api.telegram.org/bot{self.token}/getUpdates", body, None, "description")
        return found.get("result") or []


@dataclass(frozen=True)
class Texter:
    """Texts through Twilio's Messages API, from a number of the account."""

    sid: str
    token: str
    sender: str

    def send(self, to: str, text: str) -> None:
        auth = base64.b64encode(f"{self.sid}:{self.token}".encode()).decode()
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json"
        _post(url, {"To": to, "From": self.sender, "Body": text}, f"Basic {auth}", "message")


def _post(url: str, body: dict, auth: str | None, error_field: str) -> dict:
    """POST a form. An error names the service's own reason, never the URL, which holds the Telegram token."""
    request = urllib.request.Request(url, data=urllib.parse.urlencode(body).encode())
    if auth:
        request.add_header("Authorization", auth)
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            reason = json.loads(exc.read()).get(error_field)
        except ValueError:
            reason = None
        raise RuntimeError(f"HTTP {exc.code}: {reason or exc.reason}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"unreachable: {exc.reason}") from None


@dataclass(frozen=True)
class Senders:
    """Whichever of the three the secret scope has credentials for."""

    telegram: Telegram | None
    mailer: Mailer | None
    texter: Texter | None

    @classmethod
    def from_scope(cls, scope: str | None, channel: str = CHANNEL, profile: str | None = None) -> Senders:
        def secret(key):
            return read_secret(scope, key, profile)

        token = secret(TELEGRAM_SECRET)
        user, password = secret(SMTP_USER_SECRET), secret(SMTP_PASSWORD_SECRET)
        twilio = [secret(k) for k in TWILIO_SECRETS]
        return cls(
            telegram=Telegram(token, channel) if token else None,
            mailer=Mailer(user, password) if user and password else None,
            texter=Texter(*twilio) if all(twilio) else None,
        )


def deliver(senders: Senders, channel: str, address: str, email: EmailMessage, text: str) -> str:
    """Send one subscriber's message by their channel: 'sent', or why not."""
    try:
        if channel == "email":
            if senders.mailer is None:
                return f"email isn't set up: no {SMTP_USER_SECRET} / {SMTP_PASSWORD_SECRET} in the secret scope"
            senders.mailer.send(email)
        else:
            if senders.texter is None:
                return "texts aren't set up: no Twilio account in the secret scope"
            senders.texter.send(address, text)
    except Exception as exc:  # noqa: BLE001 - one bad address mustn't stop the rest
        return f"failed: {type(exc).__name__}: {exc}"[:300]
    return "sent"


# --- on Databricks ---------------------------------------------------------------------------------------------------


def run(spark, *, ml: str, app: str | None, model: str, tick, now, senders: Senders) -> dict:
    """Send what is due at `tick`, the newest tick `model` scored, log it in ml.warning_posts, and welcome new
    subscribers. Returns what happened, for the run's log."""
    out: dict = {"tick": str(tick), "model": model}
    subscriptions = f"{app}.{SUBSCRIPTIONS}" if app else None
    if subscriptions and not _ensure(spark, subscriptions, SUBSCRIPTIONS_COLUMNS, _SUBSCRIPTIONS_DOC):
        subscriptions = None
    subs = _subscribers(spark, subscriptions) if subscriptions else _no_subscribers()
    out["welcomed"] = _welcome(spark, subscriptions, subs, senders, now) if subscriptions else {}
    if tick is None or pd.isna(tick) or now - pd.Timestamp(tick) > FRESH:
        return out | {"posts": 0, "skipped": "no fresh tick"}

    posts = f"{ml}.{POSTS}"
    _ensure(spark, posts, POSTS_COLUMNS, _POSTS_DOC)
    warnings = in_force(_ratings(spark, f"{ml}.{C.PREDICTIONS}", model, tick), tick)
    sent = due(warnings, _last_posts(spark, posts))
    out["in_force"] = int(len(warnings))
    if sent.empty:
        return out | {"posts": 0}

    if senders.telegram is None:
        telegram = f"off: no {TELEGRAM_SECRET} in the secret scope"
    else:
        try:
            senders.telegram.send(channel_post(sent, tick))
            telegram = "sent"
        except Exception as exc:  # noqa: BLE001 - the subscribers still hear
            telegram = f"failed: {exc}"[:300]
    emails, texts, delivered = _notify(subs, sent, tick, senders)
    log_rows = sent.assign(
        tick=pd.Timestamp(tick),
        model=model,
        posted_at=pd.Timestamp(now),
        telegram=telegram,
        emails=sent["area"].map(emails).fillna(0).astype(int),
        texts=sent["area"].map(texts).fillna(0).astype(int),
    )
    _append(spark, posts, log_rows, POSTS_COLUMNS)
    return out | {"posts": int(len(sent)), "areas": sorted(sent["area"]), "telegram": telegram} | delivered


def _notify(subs: pd.DataFrame, sent: pd.DataFrame, tick, senders: Senders):
    """One message per subscriber for their areas among `sent`: (emails per area, texts per area, outcome counts)."""
    emails: dict[str, int] = {}
    texts: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    for sub in subs.itertuples(index=False):
        theirs = sent[sent["area"].isin(list(sub.areas))]
        if theirs.empty:
            continue
        status = deliver(
            senders, sub.channel, sub.address, email_warning(sub.address, theirs, tick), text_warning(theirs, tick)
        )
        outcomes[status.split(":")[0]] = outcomes.get(status.split(":")[0], 0) + 1
        if status == "sent":
            counts = emails if sub.channel == "email" else texts
            for area in theirs["area"]:
                counts[area] = counts.get(area, 0) + 1
    return emails, texts, {"subscribers": outcomes}


def _welcome(spark, table: str, subs: pd.DataFrame, senders: Senders, now) -> dict:
    """Welcome subscribers made in the last day who haven't had one, and record how it went on their rows."""
    waiting = subs[subs["welcomed_at"].isna() & (subs["created_at"] > pd.Timestamp(now) - WELCOME_FOR)]
    done: dict[str, int] = {}
    for sub in waiting.itertuples(index=False):
        areas = sorted(sub.areas)
        status = deliver(senders, sub.channel, sub.address, welcome_email(sub.address, areas), welcome_text(areas))
        done[status.split(":")[0]] = done.get(status.split(":")[0], 0) + 1
        if status != sub.welcome and _ID.match(sub.id):
            sent_at = "current_timestamp()" if status == "sent" else "NULL"
            spark.sql(
                f"UPDATE {table} SET welcome = {_quoted(status)}, welcomed_at = {sent_at} WHERE id = {_quoted(sub.id)}"
            )
    return done


def _ratings(spark, table: str, model: str, tick) -> pd.DataFrame:
    """`model`'s warned ratings in the HISTORY before `tick` (Low rows left out), times as naive SGT."""
    first, last = (pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in (pd.Timestamp(tick) - HISTORY, tick))
    rows = spark.sql(
        f"""SELECT area, unix_timestamp(ts) AS ts_s, tier, p_alert, reason FROM {table}
            WHERE model = {_quoted(model)} AND tier <> 'Low'
              AND ts > TIMESTAMP '{first}' AND ts <= TIMESTAMP '{last}'"""
    ).toPandas()
    return rows.assign(ts=_sgt(rows.pop("ts_s")))


def _last_posts(spark, table: str) -> pd.DataFrame:
    rows = spark.sql(
        f"""SELECT area, unix_timestamp(tick) AS tick_s, tier FROM {table}
            QUALIFY row_number() OVER (PARTITION BY area ORDER BY tick DESC, posted_at DESC) = 1"""
    ).toPandas()
    return rows.assign(tick=_sgt(rows.pop("tick_s")))


def _subscribers(spark, table: str) -> pd.DataFrame:
    rows = spark.sql(
        f"""SELECT id, channel, address, areas, welcome, unix_timestamp(created_at) AS created_s,
                   unix_timestamp(welcomed_at) AS welcomed_s
            FROM {table} WHERE active AND size(areas) > 0"""
    ).toPandas()
    return rows.assign(created_at=_sgt(rows.pop("created_s")), welcomed_at=_sgt(rows.pop("welcomed_s")))


def _no_subscribers() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "channel", "address", "areas", "welcome", "created_at", "welcomed_at"])


def _ensure(spark, table: str, columns: dict, comment: str) -> bool:
    """Create `table` if it is missing; False when its schema doesn't exist yet (the bundle makes it)."""
    spec = ", ".join(f"{c} {kind} COMMENT {_quoted(doc)}" for c, (kind, doc) in columns.items())
    try:
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({spec}) COMMENT {_quoted(comment)}")
    except Exception as exc:  # noqa: BLE001 - no schema: no subscriptions yet
        log.warning("can't use %s: %s", table, exc)
        return False
    return True


def _append(spark, table: str, frame: pd.DataFrame, columns: dict) -> None:
    rows = frame[list(columns)].astype({"tier": str}).copy()
    rows["previous"] = rows["previous"].astype(object).where(rows["previous"].notna(), None)
    schema = ", ".join(f"{c} {kind}" for c, (kind, _) in columns.items())
    spark.createDataFrame(rows, schema=schema).write.mode("append").saveAsTable(table)


def _sgt(epoch_s: pd.Series) -> pd.Series:
    """Naive SGT from epoch seconds (freshness.sgt says why times are read that way)."""
    ts = pd.to_datetime(pd.to_numeric(epoch_s), unit="s", utc=True)
    return ts.dt.tz_convert("Asia/Singapore").dt.tz_localize(None)


def _quoted(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


_POSTS_DOC = "Every warning FloodSense posted to its Telegram channel and sent to subscribers (ml/broadcast.py)"
_SUBSCRIPTIONS_DOC = (
    "People who asked the FloodSense app for warnings by email or text. Personal data: only the app and flood-live "
    "read it (ml/broadcast.py)"
)


# --- trying it out ---------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    senders = Senders.from_scope(args.secret_scope, args.channel, args.profile)
    tick = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None).floor("5min")
    sample = pd.DataFrame(
        {
            "area": ["Bedok"],
            "tier": ["High"],
            "since": [tick],
            "issued": [tick],
            "until": [tick + WARNING],
            "p_alert": [0.06],
            "reason": ["heavy rain at nearby gauges"],
            "previous": [None],
        }
    )
    status = 0
    if args.test_telegram:
        if senders.telegram is None:
            log.error("no %s in secret scope %r", TELEGRAM_SECRET, args.secret_scope)
            status = 1
        else:
            senders.telegram.send("TEST: this is how a FloodSense warning looks. Nothing is happening.\n\n"
                                  + channel_post(sample, tick))
            log.info("posted a test to %s", args.channel)
    for address in args.test_email or []:
        email = email_warning(address, sample, tick)
        email.replace_header("Subject", "[TEST] " + email["Subject"])
        log.info("email to %s: %s", address, deliver(senders, "email", address, email, ""))
    for number in args.test_sms or []:
        log.info("text to %s: %s", number, deliver(senders, "sms", number, EmailMessage(),
                                                  "TEST " + text_warning(sample, tick)))
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a test warning by each channel FloodSense uses.")
    parser.add_argument("--secret-scope", default="floodsense")
    parser.add_argument("--profile", help="Databricks CLI profile that reads the secret scope (off Databricks)")
    parser.add_argument("--channel", default=CHANNEL, help="Telegram channel the bot posts to (default %(default)s)")
    parser.add_argument("--test-telegram", action="store_true", help="post a test message to the channel")
    parser.add_argument("--test-email", nargs="*", help="email a test warning to these addresses")
    parser.add_argument("--test-sms", nargs="*", help="text a test warning to these numbers (+65...)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
