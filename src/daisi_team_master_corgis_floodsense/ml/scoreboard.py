"""The scoreboard: how every FloodSense model is judged against PUB's alerts, event by event.

Per NEA forecast area:

- A tick at Medium or above warns of a PUB alert within the next hour, so a warning stays in force until an hour after
  its last warned tick. Warned ticks at most an hour apart form one warning *episode*.
- PUB alerts at most an hour apart form one *event*, roughly one storm. Its first alert is "PUB's first alert"; a
  flood event is timed at its first flash flood.
- An event is *flagged* if an episode was already in force when it happened; lead time runs from the episode's start.
- A Medium episode with no PUB alert while it is in force is a *false alarm*. A High episode is a hit only if a flash
  flood follows while it is in force.

Training labels (tick_labels) use the same one-hour window, so an episode is a hit exactly when one of its ticks has
label 1.

Where labels are missing (posts PUB deleted before the web archive captured them), nothing is guessed: an event counts
only if every alert from an hour before PUB's first alert was recorded, since an earlier alert could be lost; a warning no alert
confirmed counts as a false alarm only if its whole watch was recorded; and missing time isn't time at risk.

Where features are missing (NEA outages, days not backfilled yet: the model is *blind*), an event counts only if the
model saw every tick that could have put a warning in force at it, from an hour before PUB's first alert; blind time
isn't time at risk either. A warning needs no such check: the model saw what it warned on.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contract as C

HORIZON = pd.Timedelta(minutes=C.HORIZON_MINUTES)
TICK = pd.Timedelta(minutes=C.TICK_MINUTES)
MONTH = pd.Timedelta(days=365.25 / 12)
_MINUTE = pd.Timedelta(minutes=1)
_HOUR = pd.Timedelta(hours=1)
_RANK = {tier: rank for rank, tier in enumerate(C.TIERS)}

# Round 1 targets. The gated LightGBM must also beat the rainfall rule on both lead time and false alarms.
TARGETS = {
    "flood_flagged_before_pub": 0.8,  # more than 80% of flash floods flagged before PUB's first alert
    "flood_mean_warning_min": 30.0,  # average warning before a flash flood, up from PUB's 13.3 min
    "high_precision": 1 / 3,  # at least 1 in 3 High warnings followed by a real flood
}


def tick_labels(
    ticks: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    kinds: Collection[str] = C.ALERT_KINDS,
    horizon: pd.Timedelta = HORIZON,
) -> pd.Series:
    """1 where a PUB alert of `kinds` comes in the tick's area within (ts, ts + horizon], else 0.

    The training label for a learned model. `ticks` needs area and ts; `alerts` is silver.area_alerts.
    """
    left = pd.DataFrame(
        {"area": _keys(ticks["area"]), "ts": _ns(ticks["ts"])}
    ).reset_index(drop=True)
    left["row"] = np.arange(len(left))
    chosen = alerts[alerts["kind"].isin(kinds)]
    right = pd.DataFrame(
        {"area": _keys(chosen["area"]), "alert_ts": _ns(chosen["alert_ts"])}
    )
    m = pd.merge_asof(
        left.sort_values("ts"),
        right.dropna().sort_values("alert_ts"),
        left_on="ts",
        right_on="alert_ts",
        by="area",
        direction="forward",
        allow_exact_matches=False,
    ).sort_values("row")
    within = m["alert_ts"] - m["ts"] <= horizon  # no later alert: NaT, so False
    return pd.Series(within.astype(int).to_numpy(), index=ticks.index)


def alert_events(alerts: pd.DataFrame, *, gap: pd.Timedelta = HORIZON) -> pd.DataFrame:
    """Group PUB alerts into events: per area, alerts at most `gap` apart are one event, roughly one storm.

    One row per event: area, first_alert (PUB's first alert), last_alert, n_alerts, first_flood (NaT if none).
    """
    a = pd.DataFrame(
        {"area": alerts["area"], "ts": _ns(alerts["alert_ts"]), "kind": alerts["kind"]}
    )
    a = a.dropna(subset=["ts"]).sort_values(["area", "ts"])
    a["flood_ts"] = a["ts"].where(a["kind"] == C.FLASH_FLOOD)
    new = a["area"].ne(a["area"].shift()) | (a["ts"].diff() > gap)
    return (
        a.groupby(new.cumsum().rename("event"))
        .agg(
            area=("area", "first"),
            first_alert=("ts", "min"),
            last_alert=("ts", "max"),
            n_alerts=("ts", "size"),
            first_flood=("flood_ts", "min"),
        )
        .reset_index(drop=True)
    )


def warning_episodes(
    predictions: pd.DataFrame,
    *,
    min_tier: str = "Medium",
    horizon: pd.Timedelta = HORIZON,
) -> pd.DataFrame:
    """Group warned ticks (tier at or above `min_tier`) into episodes: per area, warned ticks at most `horizon` apart.

    One row per episode: area, start (first warned tick), last (last warned tick), ticks, until (last + horizon).
    """
    warned = predictions["tier"].map(_RANK) >= _RANK[min_tier]
    w = pd.DataFrame({"area": predictions["area"], "ts": _ns(predictions["ts"])})
    w = w[warned].sort_values(["area", "ts"])
    new = w["area"].ne(w["area"].shift()) | (w["ts"].diff() > horizon)
    episodes = (
        w.groupby(new.cumsum().rename("episode"))
        .agg(
            area=("area", "first"),
            start=("ts", "min"),
            last=("ts", "max"),
            ticks=("ts", "size"),
        )
        .reset_index(drop=True)
    )
    episodes["until"] = episodes["last"] + horizon
    return episodes


@dataclass(frozen=True)
class Scoreboard:
    """One model over one window: a row per PUB event (`events`) and a row per warning episode (`episodes`). Rows
    with `judged` False fell where labels are missing, and count in no metric but the *_unjudged counts."""

    events: pd.DataFrame
    episodes: pd.DataFrame
    area_months: float

    def metrics(
        self, *, ci: bool = True, n_boot: int = 2000, seed: int = 0
    ) -> dict[str, float]:
        """Flat metrics, ready for mlflow.log_metrics.

        `alert_*` scores PUB events of any kind (what Medium is for), `flood_*` scores flash floods (the Round 1
        targets). With `ci`, 95% intervals come as <name>_lo and <name>_hi: Wilson for proportions, and for mean
        warning times a bootstrap over whole days, because events on one storm day aren't independent.
        """
        out: dict[str, float] = {}
        judged = self.events["judged"]
        for kind in ("alert", "flood"):
            of_kind = self.events["kind"] == kind
            ev = self.events[of_kind & judged]
            out[f"{kind}_events_unjudged"] = int((of_kind & ~judged).sum())
            lead = ev["lead_min"].fillna(0.0)  # a miss adds no warning
            # FloodSense sits on top of PUB's alerts, so the public gets whichever warned first
            warning = np.maximum(lead, ev["pub_lead_min"])
            flagged = int(ev["before_pub"].sum())
            out[f"{kind}_events"] = len(ev)
            out[f"{kind}_days"] = _sgt_day(ev["time"]).nunique()
            out[f"{kind}_flagged_before_pub"] = (
                flagged / len(ev) if len(ev) else math.nan
            )
            out[f"{kind}_mean_lead_min"] = float(lead.mean())
            out[f"{kind}_mean_pub_warning_min"] = float(ev["pub_lead_min"].mean())
            out[f"{kind}_mean_warning_min"] = float(warning.mean())
            if ci:
                lo_hi = wilson(flagged, len(ev))
                out.update(_bounds(f"{kind}_flagged_before_pub", lo_hi))
                lo_hi = _day_bootstrap(warning, _sgt_day(ev["time"]), n_boot, seed)
                out.update(_bounds(f"{kind}_mean_warning_min", lo_hi))

        judged = self.episodes["judged"]
        is_medium = self.episodes["tier"] == "Medium"
        medium = self.episodes[is_medium & judged]
        warned_hours = (medium["until"] - medium["start"]).sum() / _HOUR
        out["warnings"] = len(medium)
        out["warnings_unjudged"] = int((is_medium & ~judged).sum())
        out["false_alarms_per_area_month"] = (
            int((~medium["hit"]).sum()) / self.area_months
        )
        out["warned_hours_per_area_month"] = float(warned_hours) / self.area_months

        is_high = self.episodes["tier"] == "High"
        high = self.episodes[is_high & judged]
        hits = int(high["hit"].sum())
        out["high_warnings"] = len(high)
        out["high_warnings_unjudged"] = int((is_high & ~judged).sum())
        out["high_precision"] = hits / len(high) if len(high) else math.nan
        if ci:
            out.update(_bounds("high_precision", wilson(hits, len(high))))
        return out


def score(
    predictions: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    start,
    end,
    n_areas: int = C.N_AREAS,
    horizon: pd.Timedelta = HORIZON,
    missing=None,
    blind=None,
) -> Scoreboard:
    """Judge one model's predictions (area, ts, tier) against silver.area_alerts over the window [start, end).

    Predictions can leave out Low ticks. Events count when they happen inside the window; a warning issued near `end`
    can still be confirmed by an alert after it. `missing` lists the ticks whose labels are incomplete
    (gold.label_ticks where label_window = 'missing') and `blind` those whose features are (gold.feature_ticks where
    feature_window = 'missing'), both the same for every area; see the module docstring.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    gaps, dark = _tick_list(missing), _tick_list(blind)
    ts = _ns(predictions["ts"])
    preds = predictions.assign(ts=ts)[_within(ts, start, end)]
    medium = warning_episodes(preds, min_tier="Medium", horizon=horizon)
    high = warning_episodes(preds, min_tier="High", horizon=horizon)

    events = alert_events(alerts, gap=horizon)
    by_alert = events[_within(events["first_alert"], start, end)]
    by_flood = events[_within(events["first_flood"], start, end)]
    ev = pd.concat(
        [
            by_alert.assign(kind="alert", time=by_alert["first_alert"]),
            by_flood.assign(kind="flood", time=by_flood["first_flood"]),
        ],
        ignore_index=True,
    )
    ev["warned_from"] = _in_force(ev["area"], ev["time"], medium)
    ev["lead_min"] = (ev["time"] - ev["warned_from"]) / _MINUTE
    ev["pub_lead_min"] = (ev["time"] - ev["first_alert"]) / _MINUTE
    ev["before_pub"] = ev["warned_from"] < ev["first_alert"]  # a miss is NaT, so False
    # An event is judged only if every alert stamped from an hour before PUB's first alert up to the event is known:
    # the label windows of the ticks an hour before them, (t, t + horizon], cover exactly that. Later ticks' windows
    # reach past the event, so a gap after it mustn't count.
    reach = TICK - pd.Timedelta(1, "ns")  # the tick at or before, and the one after
    ev["judged"] = ~_any_within(
        gaps, ev["first_alert"] - horizon - reach, ev["time"] - horizon + reach, "both"
    )
    # A warned tick in [first alert - horizon, event) puts a warning in force at the event in time to count, so the
    # model must have seen all of them.
    ev["judged"] &= ~_any_within(dark, ev["first_alert"] - horizon, ev["time"], "left")

    a = pd.DataFrame(
        {"area": alerts["area"], "ts": _ns(alerts["alert_ts"]), "kind": alerts["kind"]}
    ).dropna(subset=["ts"])
    floods = a[a["kind"] == C.FLASH_FLOOD]
    medium["hit"] = _confirmed(medium, a["area"], a["ts"])
    high["hit"] = _confirmed(high, floods["area"], floods["ts"])
    episodes = pd.concat(
        [medium.assign(tier="Medium"), high.assign(tier="High")], ignore_index=True
    )
    # a confirmed warning is a hit whatever else is missing; an unconfirmed one needs its whole watch recorded
    episodes["judged"] = episodes["hit"] | ~_any_within(
        gaps, episodes["start"], episodes["last"], "both"
    )
    unknown = np.union1d(gaps, dark)  # missing or blind: not time at risk
    in_window = (_sgt_ns(pd.Series([start]))[0] <= unknown) & (
        unknown < _sgt_ns(pd.Series([end]))[0]
    )
    recorded = (end - start) - int(in_window.sum()) * TICK
    return Scoreboard(
        events=ev[
            [
                "kind",
                "area",
                "time",
                "first_alert",
                "n_alerts",
                "warned_from",
                "lead_min",
                "pub_lead_min",
                "before_pub",
                "judged",
            ]
        ],
        episodes=episodes[
            ["tier", "area", "start", "last", "until", "ticks", "hit", "judged"]
        ],
        area_months=n_areas * (recorded / MONTH),
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes in n. Unlike the normal approximation, it stays sensible at 0/n and n/n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def _in_force(area: pd.Series, t: pd.Series, episodes: pd.DataFrame) -> pd.Series:
    """Start of the episode in force at each time t in its area (started strictly before t, not expired), else NaT."""
    if t.empty or episodes.empty:
        return pd.Series(pd.NaT, index=t.index, dtype=t.dtype)
    left = pd.DataFrame({"area": _keys(area), "t": t}).reset_index(drop=True)
    left["row"] = np.arange(len(left))
    m = pd.merge_asof(
        left.sort_values("t"),
        episodes[["area", "start", "until"]]
        .assign(area=lambda e: _keys(e["area"]))
        .sort_values("start"),
        left_on="t",
        right_on="start",
        by="area",
        allow_exact_matches=False,
    ).sort_values("row")
    return pd.Series(m["start"].where(m["t"] <= m["until"]).array, index=t.index)


def _confirmed(episodes: pd.DataFrame, area: pd.Series, t: pd.Series) -> np.ndarray:
    """Whether an alert (area, t) came in each episode's area while the episode was in force."""
    started = _in_force(area, t, episodes)
    hit = {(a, s) for a, s in zip(area, started) if not pd.isna(s)}
    return np.array(
        [(a, s) in hit for a, s in zip(episodes["area"], episodes["start"])], dtype=bool
    )


def _day_bootstrap(
    values: pd.Series, days: pd.Series, n_boot: int, seed: int
) -> tuple[float, float]:
    """95% percentile interval for the mean of `values`, resampling whole days."""
    if values.empty:
        return math.nan, math.nan
    per_day = (
        pd.DataFrame({"v": values.to_numpy(dtype=float), "day": days.to_numpy()})
        .groupby("day")["v"]
        .agg(["sum", "count"])
    )
    pick = np.random.default_rng(seed).integers(0, len(per_day), (n_boot, len(per_day)))
    sums, counts = per_day["sum"].to_numpy(), per_day["count"].to_numpy()
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


_NAT = np.datetime64("NaT", "ns").astype(np.int64)


def _tick_list(ticks) -> np.ndarray:
    """Sorted SGT nanoseconds of a list of ticks (None for none), NaT dropped."""
    ns = _sgt_ns(pd.Series(ticks if ticks is not None else [], dtype=object))
    return np.sort(ns[ns != _NAT])


def _sgt_ns(ts: pd.Series) -> np.ndarray:
    """Nanoseconds of naive SGT time, so aware and naive timestamps compare (the contract reads naive ones as SGT)."""
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    return ts.dt.as_unit("ns").to_numpy().astype(np.int64)


def _any_within(
    gaps: np.ndarray, lo: pd.Series, hi: pd.Series, closed: str
) -> np.ndarray:
    """Per row, whether a sorted gap tick lies in [lo, hi) (closed='left') or [lo, hi] (closed='both')."""
    if not len(gaps) or lo.empty:
        return np.zeros(len(lo), dtype=bool)
    first = np.searchsorted(gaps, _sgt_ns(lo), side="left")
    last = np.searchsorted(
        gaps, _sgt_ns(hi), side="right" if closed == "both" else "left"
    )
    return last > first


def _bounds(name: str, lo_hi: tuple[float, float]) -> dict[str, float]:
    return {f"{name}_lo": lo_hi[0], f"{name}_hi": lo_hi[1]}


def _within(t: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (t >= start) & (t < end)


def _sgt_day(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Singapore")
    return ts.dt.normalize()


def _keys(area: pd.Series) -> pd.Series:
    """Areas as plain objects: merge_asof needs its `by` keys of one dtype, and callers pass str, object or a category
    (train.py loads areas as a category to fit ten million rows in memory)."""
    return area.astype(object)


def _ns(ts: pd.Series) -> pd.Series:
    """Datetimes at nanosecond resolution: merge_asof refuses keys of different units, which pandas 2+ can infer."""
    ts = pd.to_datetime(ts)
    return ts.dt.as_unit("ns") if hasattr(ts.dt, "as_unit") else ts
