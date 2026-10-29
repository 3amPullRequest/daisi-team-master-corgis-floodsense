"""The live fallback for Telegram posts no pattern reads: ask Jev once, land what it answered, email the developers.

flood-live's fetch calls `Fallback.handle` with the posts each tick lands (ingest/live.py). Every post is parsed with
labels/alert_text.py, as silver will parse it, and a post the patterns can't read (`jev.needs_fallback`: no format
matched, or one matched but found no place) gets two things:

- One Jev call per post and text. `call` asks Jev (labels/jev.py) which kind of post it is and which of its phrases
  are places, through Unity Gateway as the job's own Databricks identity, and lands the call as one line in
  `telegram_jev_parses/live/` (bronze.telegram_jev_parses): when, the post, why the patterns failed, the candidate
  places, Jev's answers, its token usage and latency, or the error. That folder is the record of every use of Jev, and
  silver reads Jev's answers from it, so no call is ever repeated: `_state/telegram_jev.json` keeps the keys
  (message id and text hash) already asked. A failed call is not retried; the post waits in silver.telegram_review,
  and gold counts the label windows it could hold as missing ('post not read'). A workspace with no Jev endpoint
  (its preview off, which costs nothing) is such a failure: `JevUnavailable`.
- An email (ingest/notify.py) to the --alert-to addresses with the post's full text and Jev's reading. Sending is
  retried on later ticks for up to MAX_EMAIL_ATTEMPTS, and each failure logs the whole email as a WARNING, so the
  job's log holds it even when no SMTP account is set up.

Nothing here can fail the tick: a failure is logged and counted in the heartbeat's Telegram result (`fallback`).
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from ..labels import alert_text as T
from ..labels import jev
from . import sources as S
from .cli import read_secret
from .notify import SMTP_PASSWORD_SECRET, SMTP_USER_SECRET, Mailer, parser_alert
from .sinks import Sink, read_json, write_json

log = logging.getLogger(__name__)

STATE = "telegram_jev"
MAX_EMAIL_ATTEMPTS = 12  # an hour of 5-minute ticks
KEEP_ASKED = 1000  # keys of posts already asked, newest kept
ASK_TIMEOUT_S = 30


class JevUnavailable(RuntimeError):
    """The workspace has no valid Jev endpoint (Unity Gateway answers 404 ENDPOINT_NOT_FOUND): a Jev error."""


@dataclass
class Fallback:
    """Who to ask (Jev: a request body in, the response JSON out; None turns Jev off), and who to tell."""

    ask: Callable[[dict], dict] | None
    mailer: Mailer | None
    recipients: tuple[str, ...]
    model: str = jev.MODEL

    def handle(
        self,
        sink: Sink,
        posts: list[dict],
        *,
        started: datetime,
        runner: str,
        purpose: str = "live",
    ) -> dict:
        """Ask Jev about each post the patterns can't read, land the calls, and email (or retry emailing) alerts."""
        state = read_json(sink, S.state_path(STATE))
        asked, unsent = state.get("asked", {}), state.get("unsent", [])
        unread, records = 0, []
        for post in posts:
            parsed = T.parse(post.get("text"), service=bool(post.get("service")))
            if not jev.needs_fallback(parsed):
                continue
            unread += 1
            key = f"{post['message_id']}:{jev.text_sha256(post.get('text'))}"
            if key in asked:
                continue
            asked[key] = S.iso(started)
            record = self.call(post, parsed, runner=runner, purpose=purpose)
            if record["status"] != "skipped":
                records.append(record)
            unsent.append({"record": record, "attempts": 0})
        if records:
            sink.write_bytes(
                S.run_path(S.TELEGRAM_JEV, purpose, started), S.jsonl(records)
            )
        emailed, unsent = self._send(unsent)
        if records or emailed or unsent or state.get("unsent"):
            newest = sorted(asked.items(), key=lambda kv: kv[1])[-KEEP_ASKED:]
            write_json(
                sink, S.state_path(STATE), {"asked": dict(newest), "unsent": unsent}
            )
        return {
            "unread": unread,
            "jev_calls": len(records),
            "jev_errors": sum(r["status"] == "error" for r in records),
            "emailed": emailed,
            "email_pending": len(unsent),
        }

    def call(self, post: dict, parsed: T.Parsed, *, runner: str, purpose: str) -> dict:
        """One Jev call about one post, as the line landed in telegram_jev_parses."""
        text = post.get("text") or ""
        found = jev.candidates(text)
        record = S.record(
            S.TELEGRAM_JEV,
            fetched_at=S.utcnow(),
            runner=runner,
            purpose=purpose,
            model=self.model,
            message_id=post["message_id"],
            posted_at=post.get("posted_at"),
            text=text,
            text_sha256=jev.text_sha256(text),
            why={
                "category": parsed.category,
                "pattern": parsed.pattern,
                "problem": parsed.problem,
                "hint": parsed.hint,
            },
            candidates=[asdict(c) for c in found],
        )
        if self.ask is None:
            return {
                **record,
                "status": "skipped",
                "error": "Jev is turned off (--no-jev)",
            }
        t0 = time.monotonic()
        try:
            answer = self.ask(jev.request(text, found, self.model))
            record |= {
                "status": "ok",
                "answers": answer.get("answers"),
                "model_reported": answer.get("model"),
                "usage": answer.get("usage"),
            }
        except (
            Exception
        ) as exc:  # any failure is recorded, and the post waits for review
            log.exception("Jev failed on post #%s", post["message_id"])
            record |= {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
        record["latency_ms"] = round((time.monotonic() - t0) * 1000)
        return record

    def _send(self, unsent: list[dict]) -> tuple[int, list[dict]]:
        """Email each pending alert; what failed stays pending until MAX_EMAIL_ATTEMPTS."""
        if unsent and not self.recipients:
            log.warning(
                "%d parser alert(s) with no --alert-to address to send them to",
                len(unsent),
            )
            return 0, []
        sent, pending = 0, []
        for item in unsent:
            message = parser_alert(item["record"], list(self.recipients))
            post_id = item["record"].get("message_id")
            try:
                if self.mailer is None:
                    raise RuntimeError(
                        f"no SMTP account in the secret scope ({SMTP_USER_SECRET}, {SMTP_PASSWORD_SECRET})"
                    )
                self.mailer.send(message)
                sent += 1
                log.warning(
                    "parser alert for Telegram post #%s emailed to %s",
                    post_id,
                    message["To"],
                )
            except Exception as exc:  # noqa: BLE001 - kept, retried next tick
                attempts = item["attempts"] + 1
                log.warning(
                    "parser alert for Telegram post #%s not emailed (attempt %d of %d: %s). It reads:\n%s",
                    post_id,
                    attempts,
                    MAX_EMAIL_ATTEMPTS,
                    exc,
                    message.get_content(),
                )
                if attempts < MAX_EMAIL_ATTEMPTS:
                    pending.append({**item, "attempts": attempts})
        return sent, pending


def jev_caller(profile: str | None = None, client=None) -> Callable[[dict], dict]:
    """Jev through Unity Gateway, as the job's identity on Databricks or as `profile` elsewhere. The workspace client is
    made on the first call, so a tick without a fallback never authenticates."""
    import requests

    made = {}

    def ask(body: dict) -> dict:
        if "client" not in made:
            from databricks.sdk import WorkspaceClient

            made["client"] = client or (
                WorkspaceClient(profile=profile) if profile else WorkspaceClient()
            )
        config = made["client"].config
        resp = requests.post(
            config.host.rstrip("/") + jev.PATH,
            json=body,
            headers={**config.authenticate(), "Content-Type": "application/json"},
            timeout=ASK_TIMEOUT_S,
        )
        if resp.status_code == 404 and "ENDPOINT_NOT_FOUND" in resp.text:
            raise JevUnavailable(
                f"no valid endpoint for {body.get('model')} in this workspace (HTTP 404 ENDPOINT_NOT_FOUND; "
                "the TypeSafe AI preview is off)"
            )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    return ask


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alert-to",
        default=os.environ.get("FLOODSENSE_ALERT_TO", ""),
        help="comma-separated addresses to email when a Telegram post matches no pattern (env FLOODSENSE_ALERT_TO)",
    )
    parser.add_argument(
        "--no-jev",
        action="store_true",
        help="don't ask Jev about posts no pattern reads (still email them)",
    )
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=465)


def from_args(args: argparse.Namespace, sink: Sink) -> Fallback:
    """The fallback a fetch runs with: Jev unless --no-jev, and a mailer when --alert-to is set and the secret scope
    holds an SMTP account."""
    recipients = tuple(a.strip() for a in args.alert_to.split(",") if a.strip())
    mailer = None
    if recipients:
        user = read_secret(args.secret_scope, SMTP_USER_SECRET, args.profile)
        password = read_secret(args.secret_scope, SMTP_PASSWORD_SECRET, args.profile)
        if user and password:
            mailer = Mailer(user, password, args.smtp_host, args.smtp_port)
        else:
            log.info(
                "parser alerts: no SMTP account in secret scope %r, so they are logged, not emailed",
                args.secret_scope,
            )
    ask = (
        None if args.no_jev else jev_caller(args.profile, getattr(sink, "client", None))
    )
    return Fallback(ask, mailer, recipients)


# --- check_parser_alert.py: the alert end to end, on a made-up post ------------------------------------------------

# A made-up warning in a format PUB has never used (another bullet, a new headline); no pattern reads it.
CHECK_POST = {
    "message_id": 0,  # not a real post
    "posted_at": "2026-10-23T07:32:10+00:00",
    "service": False,
    "text": "🌧️ FLASH FLOOD RISK\n\nWater in drains has reached 90% at these locations. Avoid them for the next "
    "hour:\n▶ Jalan Boon Lay (Bus stop B21339) [15:27 hours]\n▶ Upp Paya Lebar Rd [15:31 hours]\n\nStay safe!",
}
CHECK_EXPECTED = (
    T.WARNING,
    (
        T.Located("Jalan Boon Lay (Bus stop B21339)", "15:27"),
        T.Located("Upp Paya Lebar Rd", "15:31"),
    ),
)


def check(argv: list[str] | None = None) -> int:
    """Send a test parser alert: ask Jev about CHECK_POST as a live tick would, land the call under
    telegram_jev_parses/test/, and email the alert (marked TEST) to --email-to. Fails when the email can't be sent or
    Jev answers but misreads the post; Jev failing (no endpoint included) is reported in the email, as live."""
    from .cli import add_common_args, setup

    parser = argparse.ArgumentParser(description=check.__doc__.split(";")[0])
    add_common_args(parser)
    parser.add_argument(
        "--email-to", default="", help="comma-separated addresses for the test email"
    )
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=465)
    args = parser.parse_args(argv)
    _, sink = setup(args, share=0.1)
    started = S.utcnow()
    parsed = T.parse(CHECK_POST["text"])
    fallback = Fallback(
        jev_caller(args.profile, getattr(sink, "client", None)), None, ()
    )
    record = fallback.call(CHECK_POST, parsed, runner=args.runner, purpose="test")
    path = S.run_path(S.TELEGRAM_JEV, "test", started)
    sink.write_bytes(path, S.jsonl([record]))
    log.info("landed the call: %s", path)
    if record["status"] != "ok":
        log.warning("Jev failed, as the email will say: %s", record["error"])
        ok = True
    else:
        found = [jev.Candidate(**c) for c in record["candidates"]]
        reading = jev.read(CHECK_POST["text"], found, record["answers"])
        log.info("Jev answered: %s", jev.summary(found, record["answers"]))
        log.info("usage %s, %d ms", record.get("usage"), record["latency_ms"])
        ok = (reading.category, reading.locations) == CHECK_EXPECTED
        log.log(
            logging.INFO if ok else logging.ERROR,
            "reading %s: %s %s",
            "as expected" if ok else "WRONG",
            reading.category,
            [(loc.location, loc.stamp) for loc in reading.locations],
        )
    recipients = [a.strip() for a in args.email_to.split(",") if a.strip()]
    if recipients:
        user = read_secret(args.secret_scope, SMTP_USER_SECRET, args.profile)
        password = read_secret(args.secret_scope, SMTP_PASSWORD_SECRET, args.profile)
        if not (user and password):
            log.error("no SMTP account in secret scope %r", args.secret_scope)
            return 1
        try:
            Mailer(user, password, args.smtp_host, args.smtp_port).send(
                parser_alert(record, recipients)
            )
        except (ValueError, OSError, smtplib.SMTPException) as exc:
            log.error(
                "the email wasn't sent (%s: %s). %r must hold the full Gmail address and %r its 16-letter app "
                "password (Google Account > Security > App passwords)",
                type(exc).__name__,
                exc,
                SMTP_USER_SECRET,
                SMTP_PASSWORD_SECRET,
            )
            return 1
        log.info("test email sent to %s", ", ".join(recipients))
    return 0 if ok else 1
