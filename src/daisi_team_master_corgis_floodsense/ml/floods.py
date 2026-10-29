"""What a model said before each flash flood: one row per flood, with its lead time and its tier and chance.

The scoreboard judges floods in aggregate (flood_flagged_before_pub, flood_mean_lead_min); this lays them out one by
one, so a person can see what the model showed before each real flood. `flood_table` takes a model's predictions for
every tick it scored (Low ticks included, since their chance matters too) and PUB's alerts, and gives, per flash-flood
event (scoreboard.alert_events, timed at its first flash flood):

- the lead time, exactly as scoreboard.score counts it: from the start of the Medium warning in force at the flood to
  the flood; NaN where no warning was in force (a miss). `before_pub` says whether that warning came before PUB's
  first alert of the storm, and `public_warning_min` is the warning the public got from whichever came first;
- the tier and chance at the flood: the prediction at the last tick before it, whose hour holds the flood;
- the peak tier and chance over the hour before the flood (its HORIZON / TICK = 12 ticks), and how many of those ticks
  were scored at all. A tick with no prediction is unknown, not Low: an unscored flood has NaN everywhere;
- `judged`, as the scoreboard decides it from the missing and blind ticks: whether the flood counts in the metrics.

train.py logs it out of fold for the development floods (oof_floods.json), and verify.py for the registered model on
every flood, so the two can be read side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoreboard import HORIZON, TICK, score
from .split import _sgt

TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}
TIER_NAME = {v: k for k, v in TIER_RANK.items()}
COLUMNS = [
    "area",
    "flood_ts",
    "first_alert",
    "pub_lead_min",
    "judged",
    "warned_from",
    "lead_min",
    "before_pub",
    "public_warning_min",
    "tier_at_flood",
    "p_alert_at_flood",
    "peak_tier",
    "peak_p_alert",
    "ticks_scored",
]


def hour_before(flood_ts: pd.Timestamp, horizon: pd.Timedelta = HORIZON) -> pd.DatetimeIndex:
    """The ticks whose label window holds a flood at `flood_ts`: every tick t with flood_ts - horizon <= t < flood_ts,
    newest last."""
    last = (pd.Timestamp(flood_ts) - pd.Timedelta(1, "ns")).floor(TICK)
    return pd.date_range(end=last, periods=int(horizon / TICK), freq=TICK)


def flood_table(
    predictions: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    start,
    end,
    missing=None,
    blind=None,
    horizon: pd.Timedelta = HORIZON,
) -> pd.DataFrame:
    """One row per flash-flood event in [start, end) (module docstring). `predictions` holds area, ts, tier and p_alert
    for every tick the model scored; `alerts` is silver.area_alerts; `missing` and `blind` as for scoreboard.score."""
    preds = pd.DataFrame(
        {
            "area": predictions["area"].astype(object).to_numpy(),
            "ts": _sgt(predictions["ts"]).dt.as_unit("ns").to_numpy(),
            "tier": predictions["tier"].to_numpy(),
            "p_alert": pd.to_numeric(predictions["p_alert"]).to_numpy(dtype=float),
        }
    )
    board = score(
        preds[preds["tier"] != "Low"],
        alerts,
        start=start,
        end=end,
        missing=missing,
        blind=blind,
        horizon=horizon,
    )
    ev = board.events[board.events["kind"] == "flood"].reset_index(drop=True)
    table = pd.DataFrame(
        {
            "area": ev["area"].astype(object),
            "flood_ts": ev["time"],
            "first_alert": ev["first_alert"],
            "pub_lead_min": ev["pub_lead_min"],
            "judged": ev["judged"].astype(bool),
            "warned_from": ev["warned_from"],
            "lead_min": ev["lead_min"],
            "before_pub": ev["before_pub"].astype(bool),
            # FloodSense sits on top of PUB's alerts: the public gets whichever warned first (scoreboard.metrics)
            "public_warning_min": np.maximum(ev["lead_min"].fillna(0.0), ev["pub_lead_min"]),
        }
    )

    # the 12 ticks before each flood, and what the model said at each (a merge on just those keys, so ten million
    # out-of-fold rows cost one isin)
    keys = pd.DataFrame(
        [
            (i, area, tick, k)
            for i, (area, flood) in enumerate(zip(table["area"], table["flood_ts"]))
            for k, tick in enumerate(hour_before(flood, horizon))
        ],
        columns=["flood", "area", "ts", "k"],
    )
    near = preds[preds["ts"].isin(keys["ts"].unique())] if len(keys) else preds.iloc[:0]
    said = keys.merge(near, on=["area", "ts"], how="left")
    said["rank"] = said["tier"].map(TIER_RANK)
    last_k = int(horizon / TICK) - 1
    at_flood = said[said["k"] == last_k].set_index("flood")
    scored = said.dropna(subset=["rank"]).groupby("flood")
    table["tier_at_flood"] = at_flood["tier"].reindex(table.index)
    table["p_alert_at_flood"] = at_flood["p_alert"].reindex(table.index)
    table["peak_tier"] = scored["rank"].max().map(TIER_NAME).reindex(table.index)
    table["peak_p_alert"] = scored["p_alert"].max().reindex(table.index)
    table["ticks_scored"] = scored.size().reindex(table.index).fillna(0).astype(int)
    return table[COLUMNS].sort_values("flood_ts", kind="stable").reset_index(drop=True)


def summary(table: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    """Counts for MLflow over the judged floods: how many, how many any warning was in force at, how many were warned
    before PUB, and their mean lead (misses as 0) and peak chance."""
    judged = table[table["judged"]]
    lead = judged["lead_min"].fillna(0.0)
    return {
        f"{prefix}floods": float(len(table)),
        f"{prefix}floods_judged": float(len(judged)),
        f"{prefix}floods_warned": float(judged["lead_min"].notna().sum()),
        f"{prefix}floods_warned_before_pub": float(judged["before_pub"].sum()),
        f"{prefix}floods_mean_lead_min": float(lead.mean()) if len(judged) else float("nan"),
        f"{prefix}floods_median_peak_p_alert": float(judged["peak_p_alert"].median())
        if judged["peak_p_alert"].notna().any()
        else float("nan"),
    }
