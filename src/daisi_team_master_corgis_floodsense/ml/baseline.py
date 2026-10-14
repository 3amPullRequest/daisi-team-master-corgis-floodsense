"""The rainfall-rule baseline: warn when the heaviest recent rain near an area crosses a threshold.

It is the bar the gated LightGBM has to clear on lead time and false alarms (Round 1 proposal). Its training curve
also shows where the gate can go: the highest threshold that still flags nearly every training alert, and the share
of area-ticks that threshold would pass to the second model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contract as C
from .scoreboard import TARGETS, TICK, score

THRESHOLDS_MM = tuple(2.5 * k for k in range(1, 49))  # 2.5 to 120 mm


@dataclass(frozen=True)
class RainfallRule:
    """Medium from `medium_mm` and High from `high_mm`, both on the `window_min`-minute rainfall column."""

    window_min: int
    medium_mm: float
    high_mm: float = math.inf

    @property
    def column(self) -> str:
        return C.rain_column(self.window_min)

    @property
    def name(self) -> str:
        return f"rainfall_rule {self.column} M>={self.medium_mm:g} H>={self.high_mm:g}"

    def tiers(self, features: pd.DataFrame) -> np.ndarray:
        """A tier per row of gold.area_features. Missing rain reads as no rain, so Low."""
        rain = features[self.column]
        return np.select(
            [rain >= self.high_mm, rain >= self.medium_mm], ["High", "Medium"], "Low"
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Rows for ml.predictions, with the reason the app shows for every warning."""
        tier = self.tiers(features)
        warned = tier != "Low"
        reason = pd.Series(None, index=features.index, dtype=object)
        reason[warned] = [
            f"{mm:.0f} mm of rain within {C.GAUGE_RADIUS_KM:g} km in the last {self.window_min} min"
            for mm in features.loc[warned, self.column]
        ]
        return pd.DataFrame(
            {
                "model": self.name,
                "area": features["area"],
                "ts": features["ts"],
                "tier": tier,
                "p_alert": math.nan,
                "reason": reason,
            }
        )


def fit(
    features: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    start,
    end,
    n_areas: int = C.N_AREAS,
    windows: tuple[int, ...] = C.RAIN_WINDOWS_MINUTES,
    thresholds: tuple[float, ...] = THRESHOLDS_MM,
    max_false_alarms: float = 1.0,
    min_high_precision: float = TARGETS["high_precision"],
) -> tuple[RainfallRule, pd.DataFrame]:
    """Choose the rule on the training window [start, end).

    Medium: the window and threshold that flag the most PUB alert events before they happen while raising at most
    `max_false_alarms` per area per month. Ties go to the longer mean lead, then to fewer warned hours.
    High: the lowest higher threshold whose High warnings are followed by a flash flood at least `min_high_precision`
    of the time. If no threshold gets there, the rule never says High.

    `features` (gold.area_features) can leave out ticks without rain, since they are Low under every threshold.
    Returns the rule and every Medium candidate's training scores.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    features = features[(features["ts"] >= start) & (features["ts"] < end)]
    area_ticks = n_areas * ((end - start) / TICK)

    def scores(rule: RainfallRule) -> dict[str, float]:
        preds = features[["area", "ts"]].assign(tier=rule.tiers(features))
        board = score(preds, alerts, start=start, end=end, n_areas=n_areas)
        return board.metrics(ci=False)

    rows = []
    for window in windows:
        rain = features[C.rain_column(window)]
        for mm in thresholds:
            m = scores(RainfallRule(window, mm))
            if not m["alert_events"]:
                raise ValueError("no PUB alerts in the training window")
            rows.append(
                {
                    "window_min": window,
                    "medium_mm": mm,
                    "flagged": m["alert_flagged_before_pub"],
                    "lead_min": m["alert_mean_lead_min"],
                    "false_alarms": m["false_alarms_per_area_month"],
                    "warned_hours": m["warned_hours_per_area_month"],
                    "tick_share": int((rain >= mm).sum()) / area_ticks,
                }
            )
    curve = pd.DataFrame(rows)

    within = curve[curve["false_alarms"] <= max_false_alarms]
    if within.empty:
        raise ValueError(
            f"every candidate raises more than {max_false_alarms} false alarms per area-month; "
            "try higher thresholds or a bigger budget"
        )
    best = within.sort_values(
        ["flagged", "lead_min", "warned_hours"], ascending=[False, False, True]
    ).iloc[0]
    window, medium_mm = int(best["window_min"]), float(best["medium_mm"])

    high_mm = math.inf
    for mm in sorted(t for t in thresholds if t > medium_mm):
        m = scores(RainfallRule(window, medium_mm, mm))
        if m["high_warnings"] and m["high_precision"] >= min_high_precision:
            high_mm = mm
            break
    return RainfallRule(window, medium_mm, high_mm), curve
