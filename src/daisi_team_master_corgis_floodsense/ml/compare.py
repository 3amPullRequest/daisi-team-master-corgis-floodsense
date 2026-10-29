"""What a group of features adds: the gated pair fitted with and without it, compared on the same out-of-fold
scoreboard (train.py --compare-without), and any fit of a sweep against the sweep's first (train.py's vs_first_*).

The views, all over the development period and judged by scoreboard.score:
- `metric_table`: each headline metric with the group and without it, and the difference;
- `paired_events`: the PUB events judged in both fits, matched one to one. Events on one storm day aren't independent
  and both fits see the same events, so the lead difference and the flagged-before-PUB difference get a bootstrap over
  whole days, and the flagged difference an exact McNemar test on the events only one fit flagged as well;
- `paired_rank`: the same events weighed as train.py ranks fits (a flood counting flood_value times), the difference
  with a day bootstrap;
- `paired_rows`: every development row, out of fold: the difference in log loss of the alert label, which no cut-off
  chooses, so it shows signal that the Medium cut-off may not turn into events flagged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .scoreboard import Scoreboard, _day_bootstrap, _sgt_day

# headline metrics, and whether higher is better
COMPARED = {
    "alert_flagged_before_pub": True,
    "alert_mean_lead_min": True,
    "alert_mean_warning_min": True,
    "flood_flagged_before_pub": True,
    "false_alarms_per_area_month": False,
    "warned_hours_per_area_month": False,
    "high_precision": True,
    "brier": False,
    "gate_pass_share": False,
}
EVENT_KEY = ["kind", "area", "time"]


def metric_table(with_group: dict, without_group: dict) -> pd.DataFrame:
    """One row per COMPARED metric present in both: its value with the group, without it, and with minus without."""
    rows = []
    for name, higher_better in COMPARED.items():
        if name not in with_group or name not in without_group:
            continue
        a, b = float(with_group[name]), float(without_group[name])
        diff = a - b
        better = (
            "same"
            if math.isclose(a, b, abs_tol=1e-9) or math.isnan(diff)
            else ("with" if (diff > 0) == higher_better else "without")
        )
        rows.append(
            {
                "metric": name,
                "with": a,
                "without": b,
                "difference": diff,
                "better": better,
            }
        )
    return pd.DataFrame(rows)


def paired_events(
    with_group: Scoreboard,
    without_group: Scoreboard,
    kind: str = "alert",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """The events of `kind` judged in both fits: how many each flagged before PUB, how many only one did (with an exact
    McNemar p-value), the difference in the share flagged (with minus without) with a 95% day-bootstrap interval and
    two-sided p-value, and the mean lead difference (misses as 0) with a 95% day-bootstrap interval."""
    from scipy.stats import binomtest

    both = _judged_in_both(with_group, without_group, kind)
    a, b = both["before_pub_with"], both["before_pub_without"]
    flagged = day_bootstrap_difference(
        a.astype(float) - b.astype(float), _sgt_day(both["time"]), n_boot=n_boot, seed=seed
    )
    only_with, only_without = int((a & ~b).sum()), int((~a & b).sum())
    discordant = only_with + only_without
    p = (
        binomtest(only_with, discordant, 0.5).pvalue if discordant else 1.0
    )  # exact McNemar
    lead = both["lead_min_with"].fillna(0.0) - both["lead_min_without"].fillna(0.0)
    lo, hi = (
        _day_bootstrap(lead, _sgt_day(both["time"]), n_boot, seed)
        if len(both)
        else (math.nan, math.nan)
    )
    return {
        f"{kind}_events_paired": len(both),
        f"{kind}_flagged_both": int((a & b).sum()),
        f"{kind}_flagged_only_with": only_with,
        f"{kind}_flagged_only_without": only_without,
        f"{kind}_mcnemar_p": float(p),
        **{f"{kind}_flagged_difference{k}": v for k, v in flagged.items()},
        f"{kind}_lead_difference_min": float(lead.mean()) if len(both) else math.nan,
        f"{kind}_lead_difference_min_lo": float(lo),
        f"{kind}_lead_difference_min_hi": float(hi),
    }


def paired_rank(
    with_group: Scoreboard,
    without_group: Scoreboard,
    flood_value: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """The difference (with minus without) in the share of PUB events flagged before PUB with a flood counting
    `flood_value` times, as gated.weighted_flagged ranks fits, over the events judged in both: each PUB event weighs 1
    and each flood event `flood_value` - 1 more. With a 95% day-bootstrap interval and two-sided p-value."""
    weighed = [
        _judged_in_both(with_group, without_group, kind).assign(weight=weight)
        for kind, weight in (("alert", 1.0), ("flood", flood_value - 1.0))
        if weight > 0
    ]
    both = pd.concat([ev for ev in weighed if len(ev)] or weighed[:1], ignore_index=True)
    diff = both["before_pub_with"].astype(float) - both["before_pub_without"].astype(float)
    out = day_bootstrap_difference(
        diff, _sgt_day(both["time"]), both["weight"], n_boot=n_boot, seed=seed
    )
    return {f"rank_difference{k}": v for k, v in out.items()}


def paired_rows(
    p_with: np.ndarray,
    p_without: np.ndarray,
    y: np.ndarray,
    ts: pd.Series,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Two fits' out-of-fold chances of the same rows (in one order) against the alert label: the mean log loss with
    minus without (below 0: `with` is better), with a 95% day-bootstrap interval and two-sided p-value, and the rows
    and days behind it."""
    if not (len(p_with) == len(p_without) == len(y) == len(ts)):
        raise ValueError("the fits' rows differ: pair the chances of one development frame")

    def log_loss(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), LOSS_FLOOR, 1 - LOSS_FLOOR)
        return -np.where(y == 1, np.log(p), np.log1p(-p))

    diff = pd.Series(log_loss(p_with) - log_loss(p_without))
    days = _sgt_day(pd.to_datetime(pd.Series(ts).reset_index(drop=True)))
    out = day_bootstrap_difference(diff, days, n_boot=n_boot, seed=seed)
    return {
        **{f"log_loss_difference{k}": v for k, v in out.items()},
        "rows_paired": len(diff),
        "days_paired": int(days.nunique()),
    }


# chances are clipped this far from 0 and 1 in the log loss, so a row the calibration gave 0 counts finitely
LOSS_FLOOR = 1e-7


def day_bootstrap_difference(
    diff: pd.Series,
    days: pd.Series,
    weight: pd.Series | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """The (weighted) mean of paired differences, with a 95% percentile interval and a two-sided p-value from a
    bootstrap that resamples whole days: twice the smaller share of resampled means on either side of 0, at most 1.
    Keys are '' for the mean, then _lo, _hi and _p."""
    if diff.empty:
        return {"": math.nan, "_lo": math.nan, "_hi": math.nan, "_p": math.nan}
    w = np.ones(len(diff)) if weight is None else np.asarray(weight, dtype=float)
    per_day = (
        pd.DataFrame({"wd": w * diff.to_numpy(dtype=float), "w": w, "day": days.to_numpy()})
        .groupby("day")[["wd", "w"]]
        .sum()
    )
    sums, weights = per_day["wd"].to_numpy(), per_day["w"].to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(per_day), (n_boot, len(per_day)))
    means = sums[pick].sum(axis=1) / weights[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
    return {"": float(sums.sum() / weights.sum()), "_lo": float(lo), "_hi": float(hi), "_p": float(p)}


def _judged_in_both(
    with_group: Scoreboard, without_group: Scoreboard, kind: str
) -> pd.DataFrame:
    """The events of `kind` judged in both fits, one row each, with each fit's before_pub and lead_min."""

    def judged(board: Scoreboard) -> pd.DataFrame:
        ev = board.events
        return ev[(ev["kind"] == kind) & ev["judged"]][
            [*EVENT_KEY, "before_pub", "lead_min"]
        ]

    return judged(with_group).merge(
        judged(without_group), on=EVENT_KEY, suffixes=("_with", "_without")
    )


def importance_share(importance: pd.DataFrame, features) -> dict[str, float]:
    """The share of each model's total gain that went to `features`: the gate's and the rater's for a gated pair, each
    rung's for the ordinal model (the importance table's <model>_gain columns)."""
    picked = importance["input"].isin(list(features))
    return {
        f"{column}_share": float(
            importance.loc[picked, column].sum()
            / max(importance[column].sum(), np.finfo(float).tiny)
        )
        for column in importance.columns
        if column.endswith("_gain")
    }
