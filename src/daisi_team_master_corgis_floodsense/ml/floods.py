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

`report_table` lays them out per PUB report instead, which is how publish.py scores the period after training for the
app and the agencies' scoreboard: every flash flood PUB reported counts on its own (two roads flooding minutes apart in
one area are two floods), and PUB's other alerts don't enter into it. Per report:

- the warning in force at the flood, its first rating (`warned_from`) and the lead time from it, as above;
- `caught`: a warning was in force and its first rating came CAUGHT_FROM (5 minutes) to HORIZON (an hour) before the
  flood. A warning from less than 5 minutes before came as the flood happened; a flood more than an hour after the
  warning's first rating came after the hour that rating warned of;
- the tier and chance at the flood and their peak over the hour before it, as above;
- `judged`: whether the model saw every tick that could have put a warning in force at the flood, the hour before it.
  A lost PUB post can't change the warning in force at a flood that was recorded, so missing labels don't matter here.

`report_summary` counts them: floods judged, warned and caught, and the mean lead over the floods a warning was in force
at.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import contract as C
from .scoreboard import (
    HORIZON,
    TICK,
    _any_within,
    _in_force,
    _tick_list,
    _within,
    score,
    warning_episodes,
    wilson,
)
from .split import _sgt

TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}
TIER_NAME = {v: k for k, v in TIER_RANK.items()}
CAUGHT_FROM = pd.Timedelta(minutes=5)  # a warning from less than this before a flood came as the flood happened
_MINUTE = pd.Timedelta(minutes=1)
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
REPORT_COLUMNS = [
    "area",
    "flood_ts",
    "location",
    "judged",
    "warned_from",
    "lead_min",
    "caught",
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
    preds = _frame(predictions)
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
    table = table.assign(**_said_before(preds, table, horizon))
    return table[COLUMNS].sort_values("flood_ts", kind="stable").reset_index(drop=True)


def report_table(
    predictions: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    start,
    end,
    blind=None,
    horizon: pd.Timedelta = HORIZON,
) -> pd.DataFrame:
    """One row per flash flood PUB reported in [start, end), each report on its own, oldest first (module docstring).
    `predictions` as for flood_table; `alerts` is silver.area_alerts, with the place PUB named in `location` where it
    has one; `blind` as for scoreboard.score. Warnings count from the ticks in [start, end), as the scoreboard's do."""
    preds = _frame(predictions)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    warned = preds[(preds["tier"] != "Low") & _within(preds["ts"], start, end)]
    episodes = warning_episodes(warned, min_tier="Medium", horizon=horizon)
    a = alerts[alerts["kind"] == C.FLASH_FLOOD]
    reports = pd.DataFrame(
        {
            "area": a["area"].astype(object).to_numpy(),
            "flood_ts": _sgt(a["alert_ts"]).dt.as_unit("ns").to_numpy(),
            "location": a["location"].to_numpy() if "location" in a else None,
        }
    )
    reports = reports[_within(reports["flood_ts"], start, end)]
    reports = reports.sort_values(["flood_ts", "area"], kind="stable").reset_index(drop=True)
    warned_from = _in_force(reports["area"], reports["flood_ts"], episodes)
    lead = (reports["flood_ts"] - warned_from) / _MINUTE
    table = reports.assign(
        judged=~_any_within(_tick_list(blind), reports["flood_ts"] - horizon, reports["flood_ts"], "left"),
        warned_from=warned_from,
        lead_min=lead,
        caught=lead.between(CAUGHT_FROM / _MINUTE, horizon / _MINUTE),  # a miss is NaN, so False
    )
    return table.assign(**_said_before(preds, table, horizon))[REPORT_COLUMNS]


def report_summary(table: pd.DataFrame) -> dict[str, float]:
    """Counts over a report_table's judged floods, named as the scoreboard table's columns (publish.py): how many, how
    many a warning was in force at, how many were caught (and that share, with a 95% Wilson interval), and the mean
    lead over the floods a warning was in force at."""
    judged = table[table["judged"].astype(bool)]
    lead = pd.to_numeric(judged["lead_min"]).dropna()
    caught = int(judged["caught"].astype(bool).sum())
    lo, hi = wilson(caught, len(judged))
    return {
        "flash_floods": float(len(judged)),
        "floods_warned": float(len(lead)),
        "floods_caught": float(caught),
        "floods_caught_share": caught / len(judged) if len(judged) else math.nan,
        "floods_caught_share_lo": lo,
        "floods_caught_share_hi": hi,
        "flood_mean_lead_min": float(lead.mean()) if len(lead) else math.nan,
    }


def _frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """area, ts (naive SGT at nanoseconds), tier and p_alert, plain: the area as objects, whatever train.py loaded it
    as."""
    return pd.DataFrame(
        {
            "area": predictions["area"].astype(object).to_numpy(),
            "ts": _sgt(predictions["ts"]).dt.as_unit("ns").to_numpy(),
            "tier": predictions["tier"].to_numpy(),
            "p_alert": pd.to_numeric(predictions["p_alert"]).to_numpy(dtype=float),
        }
    )


def _said_before(preds: pd.DataFrame, table: pd.DataFrame, horizon: pd.Timedelta) -> pd.DataFrame:
    """What the model said in the hour before each of `table`'s floods (area, flood_ts): the tier and chance at the
    last tick before it, their peak over the hour, and how many of its ticks were scored at all."""
    # the 12 ticks before each flood, and what the model said at each (a merge on just those keys, so ten million
    # out-of-fold rows cost one isin)
    keys = pd.DataFrame(
        [
            (i, area, tick, k)
            for i, (area, flood) in enumerate(zip(table["area"], table["flood_ts"]))
            for k, tick in enumerate(hour_before(flood, horizon))
        ],
        columns=["flood", "area", "ts", "k"],
    ).astype({"ts": "datetime64[ns]"})  # typed even with no floods, so the merge below still lines up
    near = preds[preds["ts"].isin(keys["ts"].unique())] if len(keys) else preds.iloc[:0]
    said = keys.merge(near, on=["area", "ts"], how="left")
    said["rank"] = said["tier"].map(TIER_RANK)
    last_k = int(horizon / TICK) - 1
    at_flood = said[said["k"] == last_k].set_index("flood")
    scored = said.dropna(subset=["rank"]).groupby("flood")
    # positions, so a table with any index lines up
    rows = pd.RangeIndex(len(table))
    return pd.DataFrame(
        {
            "tier_at_flood": at_flood["tier"].reindex(rows).to_numpy(),
            "p_alert_at_flood": at_flood["p_alert"].reindex(rows).to_numpy(),
            "peak_tier": scored["rank"].max().map(TIER_NAME).reindex(rows).to_numpy(),
            "peak_p_alert": scored["p_alert"].max().reindex(rows).to_numpy(),
            "ticks_scored": scored.size().reindex(rows).fillna(0).astype(int).to_numpy(),
        },
        index=table.index,
    )


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
