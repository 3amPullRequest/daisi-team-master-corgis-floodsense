"""The live fallback (ingest/fallback.py) and its email (ingest/notify.py): a post no pattern reads is put to Jev once,
the call is landed, and the developers get the post by email. Jev and SMTP are fakes here."""

import json
import logging
from datetime import UTC, datetime

from daisi_team_master_corgis_floodsense.ingest import fallback as FB
from daisi_team_master_corgis_floodsense.ingest import notify
from daisi_team_master_corgis_floodsense.ingest import sources as S
from daisi_team_master_corgis_floodsense.ingest.live import run_tick
from daisi_team_master_corgis_floodsense.ingest.sinks import LocalSink
from daisi_team_master_corgis_floodsense.labels import alert_text as T
from daisi_team_master_corgis_floodsense.ml import contract as C
from tests.fakes import fake_http
from tests.test_ingest_live import radar_page, routes
from tests.test_labels_jev import NEW_FORMAT

NOW = datetime(2026, 10, 23, 7, 25, tzinfo=UTC)
TEAM = ("dev1@example.com", "dev2@example.com")
NEW_POST = {
    "message_id": 1700,
    "posted_at": "2026-10-23T07:22:05+00:00",
    "service": False,
    "text": NEW_FORMAT,
}
READABLE = {
    "message_id": 1701,
    "posted_at": "2026-10-23T07:30:00+00:00",
    "service": False,
    "text": "[Risk of Flash Floods]\n\nDue to heavy rain, please avoid this location for the next 1 hour: "
    "Jalan Boonlay (from International Road to Enterprise Road) [14:14hours]",
}


class FakeJev:
    """Answers like Jev: flash flood, and yes to the one place in NEW_FORMAT. Records every request."""

    def __init__(self, fail: bool = False):
        self.requests, self.fail = [], fail

    def __call__(self, body: dict) -> dict:
        self.requests.append(body)
        if self.fail:
            raise RuntimeError("HTTP 403: preview not enabled")
        answers = {
            "category": {
                "type": "choice",
                "choice": C.FLASH_FLOOD,
                "confidence": 0.93,
                "probabilities": {C.FLASH_FLOOD: 0.93, C.WARNING: 0.05},
            }
        }
        for key, question in body["questions"].items():
            if key != "category":
                place = (
                    '"Bukit Timah Rd (towards Clementi Rd)"' in question["instructions"]
                )
                answers[key] = {"type": "noul", "noul": 0.96 if place else 0.03}
        return {
            "model": "jev-1.14.0",
            "answers": answers,
            "usage": {"input_tokens": 900, "output_tokens": 0},
        }


class FakeMailer:
    def __init__(self, failures: int = 0):
        self.sent, self.failures = [], failures

    def send(self, message) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("connection refused")
        self.sent.append(message)


def landed_calls(tmp_path) -> list[dict]:
    folder = tmp_path / S.TELEGRAM_JEV
    return [
        json.loads(line)
        for path in sorted(folder.rglob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_a_post_no_pattern_reads_is_asked_once_landed_and_emailed(tmp_path):
    jev, mailer, sink = FakeJev(), FakeMailer(), LocalSink(tmp_path)
    fallback = FB.Fallback(jev, mailer, TEAM)
    summary = fallback.handle(sink, [NEW_POST, READABLE], started=NOW, runner="test")

    assert summary == {
        "unread": 1,
        "jev_calls": 1,
        "jev_errors": 0,
        "emailed": 1,
        "email_pending": 0,
    }
    assert len(jev.requests) == 1  # the readable post never reaches Jev
    (call,) = landed_calls(tmp_path)
    assert (tmp_path / S.TELEGRAM_JEV / "live" / "date=2026-10-23").is_dir()
    assert (
        call["status"] == "ok"
        and call["purpose"] == "live"
        and call["message_id"] == 1700
    )
    assert (
        call["why"]["category"] == T.UNMATCHED and call["usage"]["input_tokens"] == 900
    )
    assert call["text_sha256"] == FB.jev.text_sha256(NEW_FORMAT)

    (email,) = mailer.sent
    assert email["To"] == ", ".join(TEAM)
    assert "#1700" in email["Subject"] and "flash_flood" in email["Subject"]
    body = email.get_content()
    assert NEW_FORMAT in body and "https://t.me/pubfloodalerts/1700" in body
    assert "Did Jev fail? No." in body and "read_by = 'jev'" in body
    assert (
        "Jev read it as flash_flood (93% sure): Bukit Timah Rd (towards Clementi Rd) at 15:21."
        in body
    )

    # the next tick sees the same post again (say, the cursor was reset): no second call, no second email
    again = fallback.handle(sink, [NEW_POST], started=NOW, runner="test")
    assert again["jev_calls"] == 0 and again["emailed"] == 0 and len(jev.requests) == 1


def test_a_failed_call_is_landed_and_still_emailed(tmp_path):
    mailer = FakeMailer()
    FB.Fallback(FakeJev(fail=True), mailer, TEAM).handle(
        LocalSink(tmp_path), [NEW_POST], started=NOW, runner="test"
    )
    (call,) = landed_calls(tmp_path)
    assert (
        call["status"] == "error" and "403" in call["error"] and "answers" not in call
    )
    (email,) = mailer.sent
    assert (
        email["Subject"]
        == "[FloodSense] RegEx could not parse PUB post #1700; Jev failed"
    )
    body = email.get_content()
    assert NEW_FORMAT in body and "Did Jev fail? Yes." in body
    assert (
        "Jev could not be asked: RuntimeError: HTTP 403: preview not enabled." in body
    )


def test_an_email_that_fails_is_retried_on_later_ticks(tmp_path, caplog):
    sink, mailer = LocalSink(tmp_path), FakeMailer(failures=1)
    fallback = FB.Fallback(FakeJev(), mailer, TEAM)
    with caplog.at_level(logging.WARNING):
        first = fallback.handle(sink, [NEW_POST], started=NOW, runner="test")
    assert first["emailed"] == 0 and first["email_pending"] == 1
    assert NEW_FORMAT in caplog.text  # the job's log holds the whole email meanwhile
    second = fallback.handle(sink, [], started=NOW, runner="test")
    assert second == {
        "unread": 0,
        "jev_calls": 0,
        "jev_errors": 0,
        "emailed": 1,
        "email_pending": 0,
    }
    assert len(mailer.sent) == 1 and len(landed_calls(tmp_path)) == 1


def test_without_an_smtp_account_an_alert_is_logged_then_dropped(tmp_path, caplog):
    sink, fallback = LocalSink(tmp_path), FB.Fallback(FakeJev(), None, TEAM)
    with caplog.at_level(logging.WARNING):
        for _ in range(FB.MAX_EMAIL_ATTEMPTS):
            summary = fallback.handle(sink, [NEW_POST], started=NOW, runner="test")
    assert summary["email_pending"] == 0 and "no SMTP account" in caplog.text
    assert len(landed_calls(tmp_path)) == 1


def test_with_jev_off_nothing_is_landed_but_the_alert_goes_out(tmp_path):
    mailer = FakeMailer()
    summary = FB.Fallback(None, mailer, TEAM).handle(
        LocalSink(tmp_path), [NEW_POST], started=NOW, runner="test"
    )
    assert summary["jev_calls"] == 0 and summary["emailed"] == 1
    assert landed_calls(tmp_path) == []
    assert "turned off" in mailer.sent[0].get_content()


def test_a_tick_puts_new_posts_past_the_fallback(tmp_path):
    jev, mailer = FakeJev(), FakeMailer()
    http, *_ = fake_http(routes(radar_page()))
    heartbeat = run_tick(
        http,
        LocalSink(tmp_path),
        runner="test",
        now=datetime(2026, 10, 10, 10, 50, tzinfo=UTC),
        fallback=FB.Fallback(jev, mailer, TEAM),
    )
    telegram = next(r for r in heartbeat["results"] if r["source"] == S.TELEGRAM)
    # the fixture page's post 4 is an announcement no pattern reads; 1, 1630 and 1631 are read
    assert telegram["ok"] and telegram["fallback"]["unread"] == 1
    assert [r["state"]["post"][:15] for r in jev.requests] == ["Stay safe on ra"]
    assert len(mailer.sent) == 1 and "post #4;" in mailer.sent[0]["Subject"]


def test_jev_failed_when_unsure_of_the_kind_or_of_the_place():
    record = FB.Fallback(FakeJev(), None, ()).call(
        NEW_POST, T.parse(NEW_FORMAT), runner="test", purpose="live"
    )
    assert notify.jev_failure(record) is None
    unsure = json.loads(json.dumps(record))
    unsure["answers"]["category"]["confidence"] = 0.4
    assert (
        "isn't sure what kind of post it is: flash_flood at 40%"
        in notify.jev_failure(unsure)
    )
    no_place = json.loads(json.dumps(record))
    for key, answer in no_place["answers"].items():
        if key != "category":
            answer["noul"] = 0.2
    assert "read it as flash_flood, but gave no phrase" in notify.jev_failure(no_place)
    email = notify.parser_alert(no_place, list(TEAM))
    assert email["Subject"].endswith("Jev failed") and NEW_FORMAT in email.get_content()


def test_check_jev_lands_one_test_call_and_passes_on_the_right_reading(
    tmp_path, monkeypatch
):
    places = {loc.location for loc in FB.CHECK_EXPECTED[1]}

    def knows_the_post(profile=None, client=None):
        def ask(body):
            answers = {
                "category": {
                    "type": "choice",
                    "choice": C.WARNING,
                    "confidence": 0.9,
                    "probabilities": {},
                }
            }
            for key, q in body["questions"].items():
                if key != "category":
                    named = any(f'"{p}"' in q["instructions"] for p in places)
                    answers[key] = {"type": "noul", "noul": 0.9 if named else 0.1}
            return {"answers": answers, "usage": {"input_tokens": 1}}

        return ask

    monkeypatch.setattr(FB, "jev_caller", knows_the_post)
    assert T.parse(FB.CHECK_POST["text"]).category == T.UNMATCHED
    assert FB.check(["--landing", str(tmp_path), "--secret-scope", ""]) == 0
    (call,) = landed_calls(tmp_path)
    assert call["purpose"] == "test" and (tmp_path / S.TELEGRAM_JEV / "test").is_dir()


class FakeConfig:
    host = "https://workspace.example"

    def authenticate(self) -> dict:
        return {"Authorization": "Bearer not-a-real-token"}


def no_endpoint(monkeypatch):
    """requests.post answering as Unity Gateway does without the Jev preview."""
    import requests

    class Resp:
        status_code = 404
        text = '{"error_code":"ENDPOINT_NOT_FOUND","message":"The given endpoint does not exist"}'

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())


def test_no_jev_endpoint_is_a_jev_failure_and_the_email_says_so(tmp_path, monkeypatch):
    no_endpoint(monkeypatch)
    ask = FB.jev_caller(client=type("Client", (), {"config": FakeConfig()})())
    mailer = FakeMailer()
    summary = FB.Fallback(ask, mailer, TEAM, FB.jev.MODEL).handle(
        LocalSink(tmp_path), [NEW_POST], started=NOW, runner="test"
    )
    assert summary["jev_errors"] == 1 and summary["emailed"] == 1
    (call,) = landed_calls(tmp_path)
    assert call["status"] == "error" and call["error"].startswith(
        "JevUnavailable: no valid endpoint for system.ai.typesafeai-jev-1-14-0"
    )
    (email,) = mailer.sent
    body = email.get_content()
    assert email["Subject"].endswith("; Jev failed") and "Did Jev fail? Yes." in body
    assert "no valid endpoint" in body and "as missing ('post not read')" in body
    assert NEW_FORMAT in body and "TEST" not in email["Subject"]


def test_the_test_alert_is_marked_and_passes_when_jev_has_no_endpoint(
    tmp_path, monkeypatch
):
    no_endpoint(monkeypatch)
    monkeypatch.setattr(
        FB,
        "jev_caller",
        lambda profile=None, client=None, real=FB.jev_caller: real(
            client=type("Client", (), {"config": FakeConfig()})()
        ),
    )
    assert FB.check(["--landing", str(tmp_path), "--secret-scope", ""]) == 0
    (call,) = landed_calls(tmp_path)
    assert call["purpose"] == "test" and call["status"] == "error"
    email = notify.parser_alert(call, ["dev1@example.com"])
    assert email["Subject"].startswith("[FloodSense TEST]")
    assert email.get_content().startswith("This is a test of the parser alert")


def test_a_sender_that_is_not_an_address_is_refused_before_connecting():
    import pytest

    with pytest.raises(ValueError, match="full address"):
        notify.Mailer("stkinzheng", "app password").send(
            notify.parser_alert(
                {"message_id": 1, "status": "skipped", "text": "x"}, ["a@b.c"]
            )
        )


# --- a foundation model in Jev's place -----------------------------------------------------------------------------


def chat_endpoint(monkeypatch, status: int = 200, content: str | None = None):
    """requests.post answering as a foundation-model serving endpoint does, recording what it was sent."""
    import requests

    sent = []

    class Resp:
        status_code = status
        text = (
            '{"error_code":"RESOURCE_DOES_NOT_EXIST","message":"Endpoint with name not found"}'
            if status == 404
            else ""
        )

        @staticmethod
        def json():
            return {
                "model": "meta-llama-3.3-70b-instruct-121024",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}}
                ],
                "usage": {
                    "prompt_tokens": 700,
                    "completion_tokens": 90,
                    "total_tokens": 790,
                },
            }

    def post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return Resp()

    monkeypatch.setattr(requests, "post", post)
    return sent


def test_by_default_a_foundation_model_reads_the_post_in_jevs_place(
    tmp_path, monkeypatch
):
    from tests.test_labels_jev import chat_reply

    found = FB.jev.candidates(NEW_FORMAT)
    sent = chat_endpoint(
        monkeypatch,
        content=chat_reply(
            found, C.FLASH_FLOOD, {"Bukit Timah Rd (towards Clementi Rd)"}
        ),
    )
    ask = FB.jev_caller(client=type("Client", (), {"config": FakeConfig()})())
    mailer = FakeMailer()
    fallback = FB.Fallback(ask, mailer, TEAM)
    assert fallback.model == FB.jev.FOUNDATION_MODEL
    summary = fallback.handle(
        LocalSink(tmp_path), [NEW_POST], started=NOW, runner="test"
    )
    assert (
        summary["jev_calls"] == 1
        and summary["jev_errors"] == 0
        and summary["emailed"] == 1
    )

    (request,) = sent
    assert (
        request["url"]
        == f"https://workspace.example/serving-endpoints/{FB.jev.FOUNDATION_MODEL}/invocations"
    )
    assert request["headers"]["Authorization"] == "Bearer not-a-real-token"
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert T.normalize(NEW_FORMAT) in request["json"]["messages"][1]["content"]

    (call,) = landed_calls(tmp_path)
    assert call["status"] == "ok" and call["model"] == FB.jev.FOUNDATION_MODEL
    assert call["model_reported"] == "meta-llama-3.3-70b-instruct-121024"
    assert call["usage"] == {"input_tokens": 700, "output_tokens": 90}
    assert json.loads(call["reply"])["category"]["choice"] == C.FLASH_FLOOD
    assert call["answers"]["category"]["choice"] == C.FLASH_FLOOD
    # silver reads the landed call exactly as Jev's
    row = FB.jev.read_row(
        NEW_FORMAT,
        NEW_POST["posted_at"],
        json.dumps(call["candidates"]),
        json.dumps(call["answers"]),
    )
    assert row["category"] == C.FLASH_FLOOD and row["pattern"] == "jev"

    (email,) = mailer.sent
    body = email.get_content()
    assert "Did Jev fail? No." in body and "read_by = 'jev'" in body
    assert (
        "Jev read it as flash_flood (90% sure): Bukit Timah Rd (towards Clementi Rd) at 15:21."
        in body
    )
    assert f"Read by {FB.jev.FOUNDATION_MODEL}." in body


def test_a_reply_that_is_not_json_is_landed_as_an_error(tmp_path, monkeypatch):
    chat_endpoint(monkeypatch, content="Sorry, I can't tell.")
    ask = FB.jev_caller(client=type("Client", (), {"config": FakeConfig()})())
    summary = FB.Fallback(ask, FakeMailer(), TEAM).handle(
        LocalSink(tmp_path), [NEW_POST], started=NOW, runner="test"
    )
    assert summary["jev_errors"] == 1
    (call,) = landed_calls(tmp_path)
    assert call["status"] == "error" and call["error"].startswith(
        "ValueError: the reply holds no JSON object"
    )


def test_a_missing_serving_endpoint_is_a_jev_failure_too(tmp_path, monkeypatch):
    chat_endpoint(monkeypatch, status=404)
    ask = FB.jev_caller(client=type("Client", (), {"config": FakeConfig()})())
    FB.Fallback(ask, None, (), "databricks-no-such-model").handle(
        LocalSink(tmp_path), [NEW_POST], started=NOW, runner="test"
    )
    (call,) = landed_calls(tmp_path)
    assert call["error"] == (
        "JevUnavailable: no valid endpoint for databricks-no-such-model in this workspace (HTTP 404; "
        "no serving endpoint has that name)"
    )
