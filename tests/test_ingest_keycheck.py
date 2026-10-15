import pytest

from daisi_team_master_corgis_floodsense.ingest.keycheck import realtime_limit, tier_for


class FixedWindowServer:
    """data.gov.sg's limiter as documented: `limit` calls per 10-second window, reset on the window boundary. Each
    call takes `latency` seconds of the fake clock."""

    def __init__(self, limit: int, *, start: float = 0.0, latency: float = 0.04):
        self.limit, self.now, self.latency = limit, start, latency
        self.window, self.used, self.calls = None, 0, 0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def call(self) -> int:
        self.calls += 1
        window = int(self.now // 10)
        if window != self.window:
            self.window, self.used = window, 0
        self.now += self.latency
        if self.used >= self.limit:
            return 429
        self.used += 1
        return 200


@pytest.mark.parametrize(("limit", "tier"), [(6, "none"), (12, "dev"), (30, "prod")])
def test_counts_one_whole_window(limit, tier):
    # starting 9.9 s into a window: a plain burst would count the end of one window and the start of the next
    server = FixedWindowServer(limit, start=9.9)
    allowed = realtime_limit(server.call, clock=server.clock, sleep=server.sleep)
    assert allowed == limit and tier_for(allowed) == tier


def test_a_limit_beyond_the_highest_tier_reads_as_it():
    server = FixedWindowServer(1000)
    allowed = realtime_limit(
        server.call, clock=server.clock, sleep=server.sleep, max_calls=40
    )
    assert allowed == 40 and tier_for(allowed) == "prod"


def test_other_answers_stop_the_check():
    with pytest.raises(RuntimeError, match="HTTP 401"):
        realtime_limit(lambda: 401)
