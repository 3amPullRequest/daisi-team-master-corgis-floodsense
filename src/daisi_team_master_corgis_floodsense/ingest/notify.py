"""The parser alert: an email to the developers when PUB posts something no pattern in labels/alert_text.py reads.

ingest/fallback.py calls `parser_alert` with the landed Jev call (the post, why the patterns failed, Jev's answers or
its error) and sends the result with a `Mailer`. The email says two things: the message the patterns (RegEx) could
not parse, with its id, link and posting time, and whether Jev failed on it. `jev_failure` decides that: Jev failed
when it couldn't be asked (an HTTP error, or turned off), when it isn't sure enough what kind of post it is, or when
it read an alert but no phrase it gave enough weight to name the place. Otherwise the email says what Jev read, which
FloodSense now uses.

Databricks can't send this. A job's failure email says only "Workload failed, see run output for details" for a
Python task, so the post's text would never reach the inbox. The email goes out by SMTP instead (Gmail by default),
as the account whose app password is kept in the `floodsense` secret scope (see DEVELOPER_NOTES "One-time setup").
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage

from ..labels import alert_text as T
from ..labels import jev
from .sources import SGT

log = logging.getLogger(__name__)

SMTP_USER_SECRET = "parser-alert-smtp-user"
SMTP_PASSWORD_SECRET = "parser-alert-smtp-password"  # a Gmail app password
TELEGRAM_POST = "https://t.me/pubfloodalerts/{}"


@dataclass(frozen=True)
class Mailer:
    """An SMTP account that sends the parser alerts (implicit TLS, port 465)."""

    user: str
    password: str
    host: str = "smtp.gmail.com"
    port: int = 465

    def send(self, message: EmailMessage) -> None:
        if (
            self.host == "smtp.gmail.com" and "@" not in self.user
        ):  # Gmail logs in by address
            raise ValueError(
                f"secret {SMTP_USER_SECRET!r} must be the sending account's full address (name@gmail.com)"
            )
        if "From" not in message:
            message["From"] = self.user
        with smtplib.SMTP_SSL(self.host, self.port, timeout=20) as smtp:
            smtp.login(self.user, self.password)
            smtp.send_message(message)


def _sgt(iso: str | None) -> str:
    return (
        datetime.fromisoformat(iso).astimezone(SGT).strftime("%Y-%m-%d %H:%M SGT")
        if iso
        else "unknown"
    )


def _why(why: dict) -> str:
    if why.get("category") == T.UNMATCHED:
        hint = why.get("hint")
        return "no format's headline matches it" + (
            f" (its words look like a {hint} post)" if hint else ""
        )
    return f"format '{why.get('pattern')}' matched, but its reader found no location"


def reading(record: dict) -> T.Parsed | None:
    """What FloodSense reads from a landed call with Jev's cut-offs, or None when Jev gave no answers."""
    if record.get("status") != "ok":
        return None
    found = [jev.Candidate(**c) for c in record.get("candidates", [])]
    return jev.read(record.get("text"), found, record.get("answers") or {})


def jev_failure(record: dict) -> str | None:
    """Why Jev failed on the post, or None when it read it."""
    if record.get("status") == "skipped":
        return "Jev is turned off (--no-jev)"
    if record.get("status") != "ok":
        return f"Jev could not be asked: {record.get('error')}"
    read = reading(record)
    if read.category == T.UNMATCHED:
        confidence = (record["answers"].get("category") or {}).get("confidence") or 0
        return (
            f"Jev isn't sure what kind of post it is: {read.hint} at {confidence:.0%}, below the "
            f"{jev.CATEGORY_MIN:.0%} FloodSense needs"
        )
    if read.problem == T.NO_LOCATION:
        return f"Jev read it as {read.category}, but gave no phrase of it at least {jev.PLACE_MIN:.0%} as the place"
    return None


def parser_alert(record: dict, recipients: list[str]) -> EmailMessage:
    """The email about one post no pattern reads, from its landed Jev call (bronze.telegram_jev_parses)."""
    message_id = record.get("message_id")
    failure = jev_failure(record)
    if failure:
        jev_lines = [
            "Did Jev fail? Yes.",
            failure + ".",
            (
                "FloodSense has no reading of this post: it waits in silver.telegram_review, and gold counts the "
                "label windows it could hold as missing ('post not read') until a pattern or a hand label reads it."
            ),
        ]
        verdict = "Jev failed"
    else:
        read = reading(record)
        confidence = (record["answers"].get("category") or {}).get("confidence") or 0
        places = "; ".join(
            f"{loc.location} at {loc.stamp or 'the posting time'}"
            for loc in read.locations
        )
        jev_lines = [
            "Did Jev fail? No.",
            f"Jev read it as {read.category} ({confidence:.0%} sure)"
            + (f": {places}." if places else "."),
            (
                "FloodSense uses that reading until a pattern reads the post. Its alerts are labels like any "
                "other, marked read_by = 'jev' in silver.alerts and silver.area_alerts."
            ),
        ]
        verdict = f"Jev read it as {read.category}"
    test = record.get("purpose") == "test"
    body = [
        *(
            [
                "This is a test of the parser alert, on a made-up post. Nothing was posted by PUB.",
                "",
            ]
            if test
            else []
        ),
        (
            f"FloodSense's patterns (RegEx, labels/alert_text.py) could not parse PUB's Telegram post "
            f"#{message_id}: {_why(record.get('why') or {})}."
        ),
        "",
        f"Post:    {TELEGRAM_POST.format(message_id)}",
        f"Posted:  {_sgt(record.get('posted_at'))}",
        "",
        "The message:",
        "-" * 72,
        record.get("text") or "",
        "-" * 72,
        "",
        *jev_lines,
        "",
        (
            f"Sent by FloodSense's live fetch ({record.get('runner')}), {_sgt(record.get('fetched_at'))}. "
            f"Read by {record.get('model')}."
        ),
    ]
    message = EmailMessage()
    message["Subject"] = (
        f"[FloodSense{' TEST' if test else ''}] RegEx could not parse PUB post #{message_id}; {verdict}"
    )
    message["To"] = ", ".join(recipients)
    message.set_content("\n".join(body))
    return message
