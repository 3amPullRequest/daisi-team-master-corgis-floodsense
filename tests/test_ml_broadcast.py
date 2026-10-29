"""broadcast: which warnings are in force, which are news, and what goes out, checked without Spark or the network."""

import io
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import broadcast as B

DAY = "2026-09-27"


def at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY} {hhmm}")


def ratings(area: str, by_tick: dict[str, str]) -> pd.DataFrame:
    """One area's warned ratings: {"11:05": "High", ...}."""
    rows = [(area, at(t), tier, 0.06 if tier == "High" else 0.035, f"{tier} at {t}") for t, tier in by_tick.items()]
    return pd.DataFrame(rows, columns=["area", "ts", "tier", "p_alert", "reason"])


def warning(frame: pd.DataFrame, hhmm: str) -> dict | None:
    out = B.in_force(frame, at(hhmm))
    return out.iloc[0].to_dict() if len(out) else None


def test_a_rating_warns_for_the_hour_after_it_and_then_lapses():
    r = ratings("Bedok", {"11:00": "Medium"})
    assert warning(r, "10:55") is None
    assert warning(r, "11:55")["tier"] == "Medium"
    assert warning(r, "11:55")["until"] == at("12:00")
    assert warning(r, "12:00") is None


def test_elevated_holds_for_its_hour_and_moderate_follows_only_after_it():
    # Elevated at 11:10, Moderate at 11:15 and 11:50: the area stays Elevated until 12:10, then Moderate until 12:50
    r = ratings("Queenstown", {"11:05": "Medium", "11:10": "High", "11:15": "Medium", "11:50": "Medium"})
    assert warning(r, "11:05")["tier"] == "Medium"
    assert warning(r, "11:50")["tier"] == "High"
    assert warning(r, "12:05")["tier"] == "High"
    assert warning(r, "12:05")["issued"] == at("11:10")
    assert warning(r, "12:10")["tier"] == "Medium"
    assert warning(r, "12:10")["until"] == at("12:50")
    assert warning(r, "12:50") is None


def test_a_warning_starts_at_its_first_rating_across_gaps_of_up_to_an_hour():
    # 10:00 and 10:55 are 55 minutes apart: one warning. 12:00 comes 65 minutes after 10:55: a new one.
    r = ratings("Geylang", {"10:00": "Medium", "10:55": "Medium", "12:00": "High"})
    assert warning(r, "11:20")["since"] == at("10:00")
    assert warning(r, "12:00")["since"] == at("12:00")


def test_the_warning_takes_the_reason_of_the_rating_that_holds_it():
    r = ratings("Kallang", {"11:00": "High", "11:30": "Medium"})
    w = warning(r, "11:40")
    assert (w["tier"], w["reason"], w["p_alert"]) == ("High", "High at 11:00", 0.06)


def posts(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame([(a, at(t), tier) for a, t, tier in rows], columns=["area", "tick", "tier"])


def news(frame: pd.DataFrame, hhmm: str, posted: pd.DataFrame) -> list[tuple]:
    sent = B.due(B.in_force(frame, at(hhmm)), posted)
    return [(r.area, r.tier, r.previous) for r in sent.itertuples()]


def test_a_new_warning_is_news_once_and_holding_at_its_level_is_not():
    r = ratings("Bedok", {"11:00": "Medium", "11:05": "Medium"})
    assert news(r, "11:00", posts()) == [("Bedok", "Medium", None)]
    assert news(r, "11:05", posts(("Bedok", "11:00", "Medium"))) == []


def test_stepping_up_or_down_is_news_and_says_from_what():
    r = ratings("Bedok", {"11:00": "Medium", "11:10": "High", "11:30": "Medium"})
    assert news(r, "11:10", posts(("Bedok", "11:00", "Medium"))) == [("Bedok", "High", "Medium")]
    # Elevated until 12:10, then the 11:30 Moderate rating holds until 12:30
    assert news(r, "12:05", posts(("Bedok", "11:10", "High"))) == []
    assert news(r, "12:10", posts(("Bedok", "11:10", "High"))) == [("Bedok", "Medium", "High")]


def test_a_warning_after_an_earlier_one_lapsed_is_news_again():
    r = ratings("Bedok", {"09:00": "Medium", "11:00": "Medium"})
    assert news(r, "11:00", posts(("Bedok", "09:00", "Medium"))) == [("Bedok", "Medium", None)]


def sent_rows() -> pd.DataFrame:
    r = pd.concat(
        [
            ratings("Kallang", {"11:00": "Medium", "11:10": "High"}),
            ratings("Geylang", {"11:10": "High"}),
            ratings("Bedok", {"10:20": "High", "10:40": "Medium", "11:10": "Medium"}),
        ],
        ignore_index=True,
    )
    return B.due(B.in_force(r, at("11:20")), posts(("Kallang", "11:00", "Medium"), ("Bedok", "10:20", "High")))


def test_one_channel_post_lists_elevated_first_and_says_until_when():
    text = B.channel_post(sent_rows(), at("11:20"))
    assert text.startswith("FloodSense at 11:20 am")
    elevated = text.index("🔴 Elevated in Geylang until 12:10 pm")
    assert text.index("🔴 Elevated in Kallang until 12:10 pm (up from Moderate)") > elevated
    assert text.index("🟠 Moderate in Bedok until 12:10 pm (down from Elevated)") > elevated
    assert text.rstrip().endswith("Follow PUB: t.me/pubfloodalerts")


def test_a_text_for_one_area_fits_one_sms():
    r = ratings("Bukit Timah", {"11:10": "High"})
    text = B.text_warning(B.due(B.in_force(r, at("11:10")), posts()), at("11:10"))
    assert text == "FloodSense 11:10 am: Elevated in Bukit Timah until 12:10 pm. Not official: follow PUB on Telegram."
    assert len(text) <= 160


def test_an_email_names_the_areas_the_advice_and_how_to_stop():
    email = B.email_warning("someone@example.com", sent_rows(), at("11:20"))
    body = email.get_content()
    assert email["Subject"] == "FloodSense: Elevated in Geylang and Kallang, Moderate in Bedok"
    assert "What to do at Elevated:" in body and "What to do at Moderate:" in body
    assert "Why, in Geylang: High at 11:10" in body
    assert B.STOP in body


class FakeMailer:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


class FakeTexter:
    def __init__(self, fail_for=()):
        self.sent, self.fail_for = [], set(fail_for)

    def send(self, to, text):
        if to in self.fail_for:
            raise RuntimeError("HTTP 400: not a mobile number")
        self.sent.append((to, text))


def subscribers(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [(f"id-{i}", channel, address, np.array(areas)) for i, (channel, address, areas) in enumerate(rows)],
        columns=["id", "channel", "address", "areas"],
    )


def test_each_subscriber_hears_once_about_their_own_areas_and_a_failure_is_counted():
    mailer, texter = FakeMailer(), FakeTexter(fail_for={"+6580000000"})
    senders = B.Senders(telegram=None, mailer=mailer, texter=texter)
    subs = subscribers(
        ("email", "a@example.com", ["Geylang", "Kallang"]),
        ("sms", "+6591234567", ["Bedok"]),
        ("sms", "+6580000000", ["Kallang"]),
        ("email", "b@example.com", ["Tampines"]),
    )
    emails, texts, outcomes = B._notify(subs, sent_rows(), at("11:20"), senders)
    assert [m["To"] for m in mailer.sent] == ["a@example.com"]
    assert texter.sent[0][0] == "+6591234567" and "Moderate in Bedok" in texter.sent[0][1]
    assert emails == {"Kallang": 1, "Geylang": 1} and texts == {"Bedok": 1}
    assert outcomes == {"subscribers": {"sent": 2, "failed": 1}}


def test_a_channel_with_no_credentials_says_so_instead_of_sending():
    senders = B.Senders(telegram=None, mailer=None, texter=None)
    assert B.deliver(senders, "sms", "+6591234567", None, "hi").startswith("texts aren't set up")
    assert B.deliver(senders, "email", "a@example.com", None, "hi").startswith("email isn't set up")


def test_a_telegram_error_gives_its_reason_and_never_the_token(monkeypatch):
    def refuse(request, timeout):
        body = io.BytesIO(json.dumps({"ok": False, "description": "Bad Request: chat not found"}).encode())
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, body)

    monkeypatch.setattr(B.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError) as err:
        B.Telegram("123:SECRET-TOKEN", "@daisi_floodsense").send("hello")
    assert str(err.value) == "HTTP 400: Bad Request: chat not found"
    assert "SECRET" not in repr(err.value) and err.value.__cause__ is None
