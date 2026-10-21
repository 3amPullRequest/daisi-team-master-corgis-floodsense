"""The tide at Tanjong Pagar: a harmonic model (utide) fitted on UHSLC's hourly sea level, and its predictions.

UHSLC's research-quality series ends in December 2023 and NEA publishes only the current and next month of high and
low waters, so every later tide comes from the model. Its heights are metres above UHSLC's station zero; gold adds the
datum offset measured against NEA's table (metres above Chart Datum). UHSLC's zero sat 0.012 m below NEA's datum on
2026-10-20, and the fitted predictions matched NEA's high and low waters within 0.11 m and 19 minutes.

Plain pandas and numpy, so tests run without Spark. The gold pipeline runs `fit_frame` on the driver (utide's fit holds
every hour by every constituent, more than a serverless Python worker allows) and `predict_frame` with applyInPandas.
Times are epoch seconds (UTC), so no result depends on a session time zone. utide is imported only where it is used, so
silver (whose pipeline doesn't install it) can read the constants.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

LATITUDE = 1.2622  # Tanjong Pagar, UHSLC station 699
# Fit on the last 19 years of the record (2005-2023), a whole 18.6-year nodal cycle.
FIT_START = date(2005, 1, 1)
MISSING_MM = -32767  # UHSLC's fill value
STEP_S = 60  # predictions are made each minute, so a high or low water is timed to the minute
PREDICT_BLOCK = 10_000  # times per utide.reconstruct call (about 11 MB of complex values at 68 constituents)

TICK, HIGH, LOW = "tick", "high", "low"
# gold.tide_datum expects each NEA high or low water within these of the model's (0.11 m and 19 min, 2026-10-20)
HEIGHT_TOLERANCE_M = 0.15
TIME_TOLERANCE_MIN = 30
FIT_COLUMNS = (
    "fit_start_s long, fit_end_s long, observations long, mean_m double, trend_m_per_year double, "
    "residual_std_m double, constituents_json string, coef_json string"
)
PREDICTION_COLUMNS = "ts_s long, height_m double, point string"

_ARRAYS = ("A", "g", "PE", "SNR", "A_ci", "g_ci")


def _datetimes(seconds) -> np.ndarray:
    return pd.to_datetime(np.asarray(seconds, dtype="int64"), unit="s").to_numpy()


def fit(seconds, heights_m):
    """utide's least-squares fit, with its default constituents, a linear trend and nodal corrections."""
    import utide

    return utide.solve(
        _datetimes(seconds),
        np.asarray(heights_m, dtype=float),
        lat=LATITUDE,
        method="ols",
        conf_int="none",
        verbose=False,
    )


def to_json(coef) -> str:
    """What `predict` needs from a fit, as JSON: the constituents' amplitudes, phases and frequencies, the mean, the
    trend and the options utide predicts with."""

    def plain(value):
        if hasattr(value, "items"):
            return {k: plain(v) for k, v in value.items() if k != "weights"}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return json.dumps(plain(coef))


def from_json(text: str):
    from utide.utilities import Bunch

    def bunch(value):
        if isinstance(value, dict):
            return Bunch({k: bunch(v) for k, v in value.items()})
        return value

    coef = bunch(json.loads(text))
    coef["name"] = np.asarray(coef["name"], dtype=object)
    for key in _ARRAYS:
        if key in coef:
            coef[key] = np.asarray(coef[key], dtype=float)
    coef.aux["frq"] = np.asarray(coef.aux["frq"], dtype=float)
    coef.aux["lind"] = np.asarray(coef.aux["lind"], dtype="int64")
    return coef


def predict(coef, seconds) -> np.ndarray:
    """Predicted heights (m above UHSLC's zero) at epoch seconds, PREDICT_BLOCK at a time: utide builds a matrix of
    every time by every constituent, and a month of minutes at once outgrew a serverless Python worker."""
    import utide

    times = _datetimes(seconds)
    return np.concatenate(
        [
            utide.reconstruct(times[i : i + PREDICT_BLOCK], coef, verbose=False).h
            for i in range(0, len(times), PREDICT_BLOCK)
        ]
        or [np.empty(0)]
    )


def extremes(seconds: np.ndarray, heights: np.ndarray) -> pd.DataFrame:
    """High and low waters of a series: its local maxima and minima (the first minute of a flat top)."""
    rising = np.diff(heights) > 0
    turn = np.flatnonzero(rising[:-1] != rising[1:]) + 1
    point = np.where(rising[turn - 1], HIGH, LOW)
    return pd.DataFrame(
        {"ts_s": seconds[turn], "height_m": heights[turn], "point": point}
    )


def constituents(coef) -> list[dict]:
    """Amplitude (m) and Greenwich phase (degrees) of each constituent, largest first."""
    order = np.argsort(-np.asarray(coef["A"]))
    return [
        {
            "name": str(coef["name"][i]),
            "amplitude_m": float(coef["A"][i]),
            "phase_deg": float(coef["g"][i]),
        }
        for i in order
    ]


# --- applyInPandas ---------------------------------------------------------------------------------------------------


def fit_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: ts_s, sea_level_m (valid hourly readings). One row: the fit, its residuals and its JSON."""
    rows = rows.dropna().sort_values("ts_s")
    coef = fit(rows["ts_s"], rows["sea_level_m"])
    residual = rows["sea_level_m"].to_numpy() - predict(coef, rows["ts_s"])
    return pd.DataFrame(
        {
            "fit_start_s": [int(rows["ts_s"].iloc[0])],
            "fit_end_s": [int(rows["ts_s"].iloc[-1])],
            "observations": [len(rows)],
            "mean_m": [float(coef["mean"])],
            "trend_m_per_year": [float(coef.get("slope", 0.0)) * 365.25],
            "residual_std_m": [float(residual.std())],
            "constituents_json": [json.dumps(constituents(coef))],
            "coef_json": [to_json(coef)],
        }
    ).astype({"constituents_json": object, "coef_json": object})


def predict_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: one row with coef_json, start_s and end_s (a span of whole ticks). Every tick's predicted height, and the
    high and low waters in [start_s, end_s), timed to the minute."""
    row = rows.iloc[0]
    start, end, tick = int(row["start_s"]), int(row["end_s"]), int(row["tick_s"])
    coef = from_json(row["coef_json"])
    # a minute either side, so a turn on the span's first or last minute is still found
    minutes = np.arange(start - STEP_S, end + STEP_S, STEP_S, dtype="int64")
    heights = predict(coef, minutes)
    turns = extremes(minutes, heights)
    turns = turns[(turns["ts_s"] >= start) & (turns["ts_s"] < end)]
    on_tick = (minutes >= start) & (minutes < end) & (minutes % tick == 0)
    ticks = pd.DataFrame(
        {"ts_s": minutes[on_tick], "height_m": heights[on_tick], "point": TICK}
    )
    out = pd.concat([ticks, turns], ignore_index=True)
    return out.astype(
        {"point": object}
    )  # not pandas 3's Arrow strings, which applyInPandas can't send back
