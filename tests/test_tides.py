"""The tide model (tides.py) on a synthetic gauge with known constituents."""

import json

import numpy as np
import pandas as pd

from daisi_team_master_corgis_floodsense import tides as T

HOUR = 3600
START_S = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp())
M2_HOURS = 12.4206012  # the principal lunar semidiurnal period
K1_HOURS = 23.9344696  # the lunisolar diurnal period


def gauge(days: int = 400, gaps: bool = True) -> pd.DataFrame:
    """Hourly readings of a tide of M2 (0.8 m) and K1 (0.3 m) about a 1.7 m mean, with a missing stretch."""
    s = START_S + np.arange(days * 24) * HOUR
    hours = (s - START_S) / HOUR
    h = (
        1.7
        + 0.8 * np.cos(2 * np.pi * hours / M2_HOURS)
        + 0.3 * np.cos(2 * np.pi * hours / K1_HOURS)
    )
    rows = pd.DataFrame({"ts_s": s, "sea_level_m": h})
    if gaps:
        rows.loc[1000:1400, "sea_level_m"] = np.nan
    return rows


def _names(columns: str) -> list[str]:
    return [c.split()[0] for c in columns.split(",")]


def test_fit_recovers_the_constituents_and_its_columns_are_the_schema():
    fitted = T.fit_frame(gauge())
    assert list(fitted.columns) == _names(T.FIT_COLUMNS)
    row = fitted.iloc[0]
    assert row["observations"] == 400 * 24 - 401
    assert abs(row["mean_m"] - 1.7) < 0.01
    # utide's nodal corrections cost a little on a tide without them
    assert row["residual_std_m"] < 0.02
    amplitude = {
        c["name"]: c["amplitude_m"] for c in json.loads(row["constituents_json"])
    }
    assert abs(amplitude["M2"] - 0.8) < 0.05 and abs(amplitude["K1"] - 0.3) < 0.05


def test_the_fit_survives_json():
    rows = gauge(gaps=False)
    coef = T.fit(rows["ts_s"], rows["sea_level_m"])
    later = START_S + 800 * 24 * HOUR + np.arange(0, 2 * 86400, 300)
    np.testing.assert_array_equal(
        T.predict(coef, later), T.predict(T.from_json(T.to_json(coef)), later)
    )


def test_extremes_are_the_turns_of_the_series():
    s = np.arange(0, 10)
    h = np.array([0, 1, 2, 2, 1, 0, -1, 0, 1, 0], dtype=float)
    turns = T.extremes(s, h)
    assert turns.to_dict("list") == {
        "ts_s": [2, 6, 8],  # a flat top's first minute
        "height_m": [2.0, -1.0, 1.0],
        "point": [T.HIGH, T.LOW, T.HIGH],
    }


def test_predictions_fill_every_tick_and_time_the_high_waters():
    rows = gauge(gaps=False)
    coef_json = T.to_json(T.fit(rows["ts_s"], rows["sea_level_m"]))
    start, end = START_S + 500 * 86400, START_S + 503 * 86400
    span = pd.DataFrame(
        {"coef_json": [coef_json], "start_s": [start], "end_s": [end], "tick_s": [300]}
    )
    out = T.predict_frame(span)
    assert list(out.columns) == _names(T.PREDICTION_COLUMNS)
    ticks = out[out["point"] == T.TICK]
    assert ticks["ts_s"].tolist() == list(range(start, end, 300))
    highs = out[out["point"] == T.HIGH]["ts_s"].to_numpy()
    assert ((highs >= start) & (highs < end)).all()
    assert highs.size in (5, 6)  # about two a day
    # every high water is the largest minute near it
    for high in highs:
        near = np.arange(high - 1800, high + 1801, 60)
        assert np.argmax(T.predict(T.from_json(coef_json), near)) == 30
