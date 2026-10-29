"""The drains' response (features.gauge_drains, contract.DRAIN_STATE) on one synthetic gauge: the recursion that gold's
gauge_rain runs per gauge in pandas."""

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense import features as F_
from daisi_team_master_corgis_floodsense.ml import contract as C

TICK_S = C.TICK_MINUTES * 60
REFERENCE = C.DRAIN_REFERENCE_MM_H / 12  # mm a tick


def drains(rain, ticks=None, shuffled: bool = False) -> pd.DataFrame:
    """gauge_drains of one gauge's arrivals, a tick each (or at `ticks`, in tick numbers), the rows in order or not."""
    ticks = np.arange(len(rain)) if ticks is None else np.asarray(ticks)
    rows = pd.DataFrame({"station_id": "S1", "tick_s": TICK_S * ticks, "mm": rain})
    return F_.gauge_drains(rows.sample(frac=1.0, random_state=1) if shuffled else rows)


def test_steady_rain_flows_out_as_it_comes_in_within_its_response_time():
    for k in C.DRAIN_RESPONSE_MINUTES:
        out = drains([REFERENCE] * 288 + [1.1 * REFERENCE] * 60)
        flow = out[f"d{k}"].to_numpy()
        # a day at the reference: what comes in flows out, less the hollows' evaporation, and nothing overflows
        assert flow[287] == pytest.approx(C.DRAIN_REFERENCE_MM_H, abs=0.3)
        assert (out[f"o{k}"][:288] == 0).all() and out[f"o{k}"].iloc[-1] > 0
        # 10% more: about 63% of the step is through after the response time, as for a time constant
        covered = (flow[287 + k // C.TICK_MINUTES] - flow[287]) / (flow[-1] - flow[287])
        assert covered == pytest.approx(1 - np.exp(-1), abs=0.05)
    # the flows come out in whole steps
    assert (out[list(F_.DRAIN_COLUMNS[:2])] * 10 % 1 == 0).all().all()


def test_without_rain_a_store_empties_as_the_reservoir_equation_says():
    # a 42 mm burst, 40 mm of it past the hollows, then dry: each tick's flow is the store's outflow over the tick,
    # against a fine Runge-Kutta integration of dS/dt = -a S^m
    m, reference = C.DRAIN_EXPONENT, C.DRAIN_REFERENCE_MM_H / 60
    out = drains([42.0] + [0.0] * 100)
    for k in C.DRAIN_RESPONSE_MINUTES:
        a = reference / (m * reference * k) ** m
        slope = lambda s: -a * max(s, 0.0) ** m  # noqa: E731
        s, h, store = 40.0, C.TICK_MINUTES / 500, [40.0]
        for _ in range(100):
            for _ in range(500):
                k1 = slope(s)
                k2 = slope(s + h * k1 / 2)
                k3 = slope(s + h * k2 / 2)
                k4 = slope(s + h * k3)
                s += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            store.append(s)
        outflow = -np.diff(store) * 60 / C.TICK_MINUTES  # mm/h over each tick
        expected = np.floor(outflow * 10 + 0.5) / 10
        flow = out[f"d{k}"].to_numpy()[:100]
        # equal but where the store's tail has been called drained (its flow rounds to 0)
        assert np.abs(flow - expected).max() <= 0.1 and (flow[expected > 0.1] == expected[expected > 0.1]).all()


def test_the_ground_takes_the_first_millimetres_and_dries_out_again():
    loss = C.DRAIN_INITIAL_LOSS_MM
    dry_ground = drains([loss] + [0.0] * 20)
    assert (dry_ground[list(F_.DRAIN_COLUMNS)] == 0).all().all()
    # the hollows dry at DRAIN_LOSS_RECOVERY_MM_DAY, so they take the next storm's first millimetres again later
    hours_to_dry = loss / C.DRAIN_LOSS_RECOVERY_MM_DAY * 24
    soon, later = int(12 * 1), int(12 * (hours_to_dry + 1))
    wet_again = drains([loss] + [0.0] * soon + [loss] + [0.0] * 5)
    dry_again = drains([loss] + [0.0] * later + [loss] + [0.0] * 5)
    assert wet_again["d10"].max() > 0
    assert dry_again["d10"].max() == 0


def test_the_overflow_is_the_flow_beyond_capacity_still_waiting():
    storm = list(25.0 * np.hanning(14)[1:-1]) + [0.0] * 60
    out = drains(storm)
    for k in C.DRAIN_RESPONSE_MINUTES:
        flow, waiting = out[f"d{k}"].to_numpy(), out[f"o{k}"].to_numpy()
        expected, level = [], 0.0
        for f in flow:
            level = max(0.0, level + (f - C.DRAIN_CAPACITY_MM_H) / 12)
            expected.append(level)
        assert waiting == pytest.approx(expected, abs=1e-9)
        assert waiting.max() > 0 and waiting[-1] == 0


def test_heavier_storms_peak_sooner_and_less_damped():
    # what a linear filter of the rain can't do, and the rain windows are
    shape = np.hanning(14)[1:-1]
    for k in C.DRAIN_RESPONSE_MINUTES:
        light = drains(list(3 * shape) + [0.0] * 40)[f"d{k}"].to_numpy()
        heavy = drains(list(10 * shape) + [0.0] * 40)[f"d{k}"].to_numpy()
        assert heavy.argmax() <= light.argmax()
        assert heavy.max() / 10 > light.max() / 3


def test_missed_ticks_are_dry_and_the_rows_order_doesnt_matter():
    rng = np.random.default_rng(0)
    rain = np.where(rng.random(600) < 0.2, rng.gamma(1.5, 3.0, 600), 0.0)
    rain[200:260] = 0.0  # the gauge was down here, so it had no rain
    full = drains(rain)
    kept = np.r_[0:200, 260:600]
    gappy = drains(rain[kept], ticks=kept, shuffled=True)
    pd.testing.assert_frame_equal(gappy, full.iloc[kept].reset_index(drop=True))
    assert gappy["station_id"].dtype == object


def test_the_stores_empty_within_the_day_live_scoring_reads():
    # live scoring's stores start empty a day back (PRIOR_RAIN_REACH): they match gold's once gold's have emptied, and
    # gold's empty within that day after even an hour at 300 mm/h
    out = drains([25.0] * 12 + [0.0] * 24 * 12)
    reach = (max(C.PRIOR_RAIN_HOURS) - 1) * 12  # ticks
    for c in F_.DRAIN_COLUMNS:
        assert out[c].max() > 0
        assert (out[c].to_numpy()[12 + reach :] == 0).all(), c
