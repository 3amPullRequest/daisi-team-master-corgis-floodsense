from datetime import date

import pytest

from daisi_team_master_corgis_floodsense.ingest.client import (
    USER_AGENT,
    HttpError,
    RateLimiter,
    dataset_file_url,
    iter_day_pages,
    realtime,
)
from tests.fakes import FakeClock, FakeResponse, api_page, fake_http, no_data


def test_rate_limiter_waits_for_the_window_to_slide():
    clock = FakeClock()
    limiter = RateLimiter(3, 10.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == []
    clock.now += 4
    limiter.acquire()  # the 4th call within 10 s waits until the 1st one is 10 s old
    assert clock.sleeps == [6.0]


def test_tier_scales_limits():
    http, *_ = fake_http([], api_key="k", share=0.6)
    assert http.tier == "dev"
    assert http.limiters["realtime"].max_calls == 7  # 12 per 10 s at 60%
    http, *_ = fake_http([], share=0.8)
    assert http.tier == "none"
    assert http.limiters["realtime"].max_calls == 4  # 6 per 10 s at 80%


def test_api_key_only_goes_to_data_gov_sg_calls():
    http, session, _ = fake_http(
        [("example", FakeResponse(200, text_body="ok"))], api_key="secret"
    )
    http.get("https://example.org/a", bucket="realtime")
    http.get("https://example.org/b")
    assert session.calls[0]["headers"] == {"x-api-key": "secret"}
    assert session.calls[1]["headers"] == {}
    assert session.headers["User-Agent"] == USER_AGENT


def test_429_is_retried_after_ten_seconds():
    busy = FakeResponse(429, {"code": 24, "errorMsg": "Rate limit exceeded"})
    http, session, clock = fake_http([("rainfall", [busy, api_page("readings", 1)])])
    resp = realtime(http, "rainfall")
    assert resp.status == 200
    assert len(session.calls) == 2
    assert 10 <= max(clock.sleeps) <= 12


def test_a_429_with_a_key_warns_once_that_the_key_may_not_be_recognised(caplog):
    busy = FakeResponse(429, {"code": 24, "errorMsg": "Rate limit exceeded"})
    pages = [busy, api_page("readings", 1), busy, api_page("readings", 1)]
    http, *_ = fake_http([("rainfall", pages)], api_key="secret")
    realtime(http, "rainfall")
    realtime(http, "rainfall")
    warnings = [r for r in caplog.records if "check_key.py" in r.getMessage()]
    assert len(warnings) == 1 and "dev-key limit" in warnings[0].getMessage()
    # without a key a 429 is just the public limit
    caplog.clear()
    http, *_ = fake_http([("rainfall", [busy, api_page("readings", 1)])])
    realtime(http, "rainfall")
    assert not [r for r in caplog.records if "check_key.py" in r.getMessage()]


def test_retry_after_header_wins():
    busy = FakeResponse(503, {"error": "down"}, headers={"Retry-After": "30"})
    http, _, clock = fake_http([("rainfall", [busy, api_page("readings", 1)])])
    realtime(http, "rainfall")
    assert 30 <= max(clock.sleeps) <= 32


def test_other_client_errors_raise():
    http, *_ = fake_http(
        [("rainfall", FakeResponse(400, {"errorMsg": "Invalid date format"}))]
    )
    with pytest.raises(HttpError):
        realtime(http, "rainfall", {"date": "bad"})


def test_no_data_404_is_a_response_but_unknown_route_raises():
    http, *_ = fake_http(
        [
            ("rainfall", no_data()),
            ("nope", FakeResponse(404, {"code": 28, "errorMsg": "Route not found"})),
        ]
    )
    assert realtime(http, "rainfall", {"date": "2016-06-01"}).no_data
    with pytest.raises(HttpError):
        realtime(http, "nope")


def test_day_pages_follow_the_token_and_stop():
    pages = [
        api_page("readings", 25, token="p2"),
        api_page("readings", 25, token="p3"),
        api_page("readings", 3),
    ]
    http, session, _ = fake_http([("rainfall", pages)])
    got = list(iter_day_pages(http, "rainfall", date(2025, 1, 1)))
    assert len(got) == 3
    assert [c["params"].get("paginationToken") for c in session.calls] == [
        None,
        "p2",
        "p3",
    ]
    assert all(c["params"]["date"] == "2025-01-01" for c in session.calls)


def test_day_pages_stop_at_no_data():
    http, *_ = fake_http([("rainfall", no_data())])
    got = list(iter_day_pages(http, "rainfall", date(2016, 6, 1)))
    assert len(got) == 1 and got[0].no_data


def test_dataset_file_url_initiates_then_polls_until_ready():
    pending = FakeResponse(201, {"code": 0, "data": {"status": "DOWNLOAD_PENDING"}})
    ready = FakeResponse(
        201,
        {"code": 0, "data": {"status": "DOWNLOAD_SUCCESS", "url": "https://s3/x.csv"}},
    )
    started = FakeResponse(
        201, {"code": 0, "data": {"message": "Download successfully initiated"}}
    )
    http, session, _ = fake_http(
        [("initiate-download", started), ("poll-download", [pending, ready])]
    )
    assert (
        dataset_file_url(http, "d_1", initiate=True, poll_every=0.1)
        == "https://s3/x.csv"
    )
    assert [c["url"].rsplit("/", 1)[1] for c in session.calls] == [
        "initiate-download",
        "poll-download",
        "poll-download",
    ]
