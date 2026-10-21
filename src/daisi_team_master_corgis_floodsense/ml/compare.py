"""What a group of features adds: the gated pair fitted with and without it, compared on the same out-of-fold
scoreboard (train.py --compare-without).

Two views, both over the development period and judged by scoreboard.score:
- `metric_table`: each headline metric with the group and without it, and the difference;
- `paired_events`: the PUB events judged in both fits, matched one to one. Events on one storm day aren't independent
  and both fits see the same events, so the lead difference gets a bootstrap over whole days and the flagged-before-PUB
  difference an exact McNemar test on the events only one fit flagged.
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
    McNemar p-value), and the mean lead difference (with minus without, misses as 0) with a 95% day-bootstrap
    interval."""
    from scipy.stats import binomtest

    def judged(board: Scoreboard) -> pd.DataFrame:
        ev = board.events
        return ev[(ev["kind"] == kind) & ev["judged"]][
            [*EVENT_KEY, "before_pub", "lead_min"]
        ]

    both = judged(with_group).merge(
        judged(without_group), on=EVENT_KEY, suffixes=("_with", "_without")
    )
    a, b = both["before_pub_with"], both["before_pub_without"]
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
        f"{kind}_lead_difference_min": float(lead.mean()) if len(both) else math.nan,
        f"{kind}_lead_difference_min_lo": float(lo),
        f"{kind}_lead_difference_min_hi": float(hi),
    }


def importance_share(importance: pd.DataFrame, features) -> dict[str, float]:
    """The share of each model's total gain that went to `features`."""
    picked = importance["input"].isin(list(features))
    return {
        f"{model}_gain_share": float(
            importance.loc[picked, f"{model}_gain"].sum()
            / max(importance[f"{model}_gain"].sum(), np.finfo(float).tiny)
        )
        for model in ("gate", "rater")
    }
