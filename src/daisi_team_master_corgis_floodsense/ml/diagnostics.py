"""A fit's out-of-fold chances beyond the scoreboard's headline: what train.py logs per fit of a sweep, as diag_* metrics
and in sweep.json (24 Oct).

- `rows_confusion`: each development row (one area-tick) warned or not, against the label of the hour after it: true
  and false positives and negatives, with the false positive rate (of the rows no alert followed), the false negative
  rate (of those one did) and the precision. A storm's hour holds 12 ticks, so these count ticks, not storms.
- `episodes_and_events`: the scoreboard's own units. Warning episodes at Medium (Medium or above) and at High, judged,
  and how many a PUB alert (for High, a flash flood) followed; PUB events and floods flagged before PUB's first alert,
  warned only after it, or missed, and how many had a High warning in force before PUB.
- `chances`: the shape of p_alert over every row and over the rows the gate passed (quantiles, skewness, kurtosis), how
  far it separates the rows an alert followed from the rest (their mean chances, ROC AUC), and its calibration: the
  alerts it expects against those that came, and the expected calibration error over deciles of the passed rows.
- `curve_budgets`: the most PUB events the fit's Medium curve flags within each of several false-alarm budgets, so a fit
  held back by the budget shows.
- `tier_table`: other ways to set the tiers, scored like the fit's own (train.py logs it for the chosen fit as
  tiers.json): Medium from each of a range of cut-offs, High from each cut-off above the fit's Medium, and High wherever
  the fit is at Medium and a condition holds (the rule's rain, say). Per definition: the PUB events and floods flagged
  before PUB, false alarms, and the warnings issued and what followed them.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .scoreboard import Scoreboard, _confirmed, _in_force, _ns, score

# the tiers that count as warned at each level
WARNED = {"medium": ("Medium", "High"), "high": ("High",)}
QUANTILES = {"p50": 0.5, "p90": 0.9, "p99": 0.99, "p99_9": 0.999, "p99_99": 0.9999}
BUDGETS = (0.5, 1.0, 2.0, 4.0)  # false alarms per area-month
DECILES = 10
# round chances tried as cut-offs in the tier table, beside the fit's own candidates
ROUND_CUT_OFFS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1)


def summary(fitted, board: Scoreboard) -> dict[str, float]:
    """Every diagnostic of one fit (gated.Fit or ordinal.Fit) and its out-of-fold scoreboard, flat."""
    rows = fitted.rows
    y = rows["alert_label"].to_numpy()
    flood = rows["flood_label"].to_numpy()
    tier = np.asarray(fitted.tier)
    out: dict[str, float] = {}
    for level, tiers in WARNED.items():
        warned = np.isin(tier, tiers)
        out |= rows_confusion(y, warned, f"{level}_alert")
        out |= rows_confusion(flood, warned, f"{level}_flood")
    out |= episodes_and_events(board)
    out |= chances(fitted.p_alert, y, getattr(fitted, "passed", None))
    out |= curve_budgets(fitted.curve)
    return out


def rows_confusion(y: np.ndarray, warned: np.ndarray, prefix: str) -> dict[str, float]:
    """True and false positives and negatives of `warned` rows against labels `y` (1 or 0), and their rates."""
    positive = y == 1
    tp = int(np.sum(warned & positive))
    fp = int(np.sum(warned & ~positive))
    fn = int(np.sum(~warned & positive))
    tn = int(np.sum(~warned & ~positive))
    return {
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_tn": tn,
        f"{prefix}_fpr": _share(fp, fp + tn),
        f"{prefix}_fnr": _share(fn, tp + fn),
        f"{prefix}_precision": _share(tp, tp + fp),
    }


def episodes_and_events(board: Scoreboard) -> dict[str, float]:
    """The judged warning episodes per tier and what followed them, and the judged PUB events by how they were warned."""
    out: dict[str, float] = {}
    judged = board.episodes[board.episodes["judged"]]
    for tier in ("Medium", "High"):
        ep = judged[judged["tier"] == tier]
        hits = int(ep["hit"].sum())
        name = tier.lower()
        out[f"{name}_episodes"] = len(ep)
        out[f"{name}_episode_hits"] = hits
        out[f"{name}_false_alarms"] = len(ep) - hits
        out[f"{name}_false_alarm_share"] = _share(len(ep) - hits, len(ep))
        out[f"{name}_warned_ticks"] = int(ep["ticks"].sum())
    high = board.episodes[board.episodes["tier"] == "High"]
    events = board.events[board.events["judged"]]
    for kind in ("alert", "flood"):
        ev = events[events["kind"] == kind]
        warned = ev["warned_from"].notna().to_numpy()
        before = ev["before_pub"].astype(bool).to_numpy()
        # a High episode in force at the event that started before PUB's first alert
        high_from = _in_force(ev["area"], ev["time"], high)
        out[f"{kind}_events_before_pub"] = int(before.sum())
        out[f"{kind}_events_late"] = int((warned & ~before).sum())
        out[f"{kind}_events_missed"] = int((~warned).sum())
        out[f"{kind}_events_high_before_pub"] = int((high_from < ev["first_alert"]).sum())
    out["area_months"] = float(board.area_months)
    return out


def chances(p_alert: np.ndarray, y: np.ndarray, passed: np.ndarray | None) -> dict[str, float]:
    """The distribution of p_alert (rows rated, NaN left out), how it separates labels, and its calibration."""
    from scipy.stats import kurtosis, skew
    from sklearn.metrics import roc_auc_score

    p = np.asarray(p_alert, dtype=float)
    rated = ~np.isnan(p)
    p, y = p[rated], y[rated]
    alert = y == 1
    out = {
        "p_mean": float(p.mean()),
        "p_std": float(p.std()),
        "p_skewness": float(skew(p)),
        "p_kurtosis": float(kurtosis(p)),
        **{f"p_{name}": float(q) for name, q in zip(QUANTILES, np.quantile(p, list(QUANTILES.values())))},
        "p_max": float(p.max()),
        "p_distinct": int(len(np.unique(p))),
        "p_mean_alert_rows": float(p[alert].mean()) if alert.any() else math.nan,
        "p_mean_quiet_rows": float(p[~alert].mean()),
        "p_median_alert_rows": float(np.median(p[alert])) if alert.any() else math.nan,
        "alert_roc_auc": float(roc_auc_score(y, p)) if alert.any() and not alert.all() else math.nan,
        # 1 when the chances add up to the alerts that came
        "expected_over_observed": _share(float(p.sum()), int(alert.sum())),
    }
    if passed is not None:
        keep = np.asarray(passed)[rated]
        pp, yp = p[keep], y[keep]
        out |= {
            "passed_share": float(keep.mean()),
            "passed_p_mean": float(pp.mean()),
            "passed_p_skewness": float(skew(pp)),
            "passed_p_kurtosis": float(kurtosis(pp)),
            "passed_expected_over_observed": _share(float(pp.sum()), int((yp == 1).sum())),
            "passed_ece": _ece(pp, yp),
        }
    return out


def curve_budgets(curve: pd.DataFrame) -> dict[str, float]:
    """The largest share of PUB events the Medium curve flags within each budget of BUDGETS, NaN where none fits."""
    out = {}
    for fa in BUDGETS:
        within = curve.loc[curve["false_alarms"] <= fa, "flagged"]
        out[f"flagged_at_{fa:g}_false_alarms"] = float(within.max()) if len(within) else math.nan
    return out


def tier_table(
    fitted,
    alerts: pd.DataFrame,
    split,
    *,
    medium_p: float,
    missing=None,
    blind=None,
    high_where: dict[str, np.ndarray] | None = None,
    candidates: int = 60,
) -> pd.DataFrame:
    """Other tier definitions for one fit, each scored over the development period as the fit's own tiers are.

    - definition 'medium_at': Medium wherever p_alert >= cut_off (no High), at the fit's candidate cut-offs (the
      quantiles cut_offs tries) and ROUND_CUT_OFFS;
    - 'high_at': the fit's Medium (`medium_p`), High wherever p_alert >= cut_off, at those above `medium_p`;
    - 'high_where': the fit's Medium, High where it is at Medium and the named mask of `high_where` (one per
      development row, in fitted.rows' order) holds.

    Columns: the PUB events and floods flagged before PUB (counts of those judged), the false alarms per area-month,
    the mean lead (misses as 0), the warned hours per area-month, the Medium warnings (judged) and how many a PUB alert
    followed, and the High warnings (judged), how many a flood followed, how many any PUB alert did, and the floods a
    High warning was in force before PUB's first alert."""
    from .gated import _candidates

    p = np.asarray(fitted.p_alert, dtype=float)
    passed = np.asarray(getattr(fitted, "passed", np.ones(len(p), dtype=bool)))
    floor = p[~passed].max() if (~passed).any() else -math.inf
    cuts = sorted({*_candidates(p[passed], candidates), *ROUND_CUT_OFFS})
    cuts = [c for c in cuts if c > floor]
    frame = fitted.rows[["area", "ts"]]
    every = pd.DataFrame({"area": alerts["area"], "ts": _ns(alerts["alert_ts"])}).dropna(subset=["ts"])

    def scored(tier: np.ndarray) -> dict[str, float]:
        warned = tier != "Low"
        board = score(
            frame[warned].assign(tier=tier[warned]),
            alerts,
            start=split.start,
            end=split.dev_end,
            missing=missing,
            blind=blind,
        )
        m = board.metrics(ci=False)
        d = episodes_and_events(board)
        high = board.episodes[(board.episodes["tier"] == "High") & board.episodes["judged"]]
        return {
            "rows_warned": int(warned.sum()),
            "events_before_pub": d["alert_events_before_pub"],
            "events_flagged_share": m["alert_flagged_before_pub"],
            "floods_before_pub": d["flood_events_before_pub"],
            "false_alarms_per_area_month": m["false_alarms_per_area_month"],
            "mean_lead_min": m["alert_mean_lead_min"],
            "warned_hours_per_area_month": m["warned_hours_per_area_month"],
            "medium_warnings": d["medium_episodes"],
            "medium_warnings_followed": d["medium_episode_hits"],
            "high_warnings": d["high_episodes"],
            "high_warnings_flood_followed": d["high_episode_hits"],
            "high_warnings_alert_followed": int(_confirmed(high, every["area"], every["ts"]).sum()) if len(high) else 0,
            "floods_high_before_pub": d["flood_events_high_before_pub"],
        }

    rows = []
    for c in cuts:
        rows.append({"definition": "medium_at", "cut_off": c, **scored(np.where(p >= c, "Medium", "Low"))})
    medium = p >= medium_p
    for c in (c for c in cuts if c > medium_p):
        tier = np.select([p >= c, medium], ["High", "Medium"], "Low")
        rows.append({"definition": "high_at", "cut_off": c, **scored(tier)})
    for name, mask in (high_where or {}).items():
        tier = np.select([medium & np.asarray(mask, dtype=bool), medium], ["High", "Medium"], "Low")
        rows.append({"definition": "high_where", "cut_off": medium_p, "condition": name, **scored(tier)})
    return pd.DataFrame(rows)


def _ece(p: np.ndarray, y: np.ndarray, bins: int = DECILES) -> float:
    """Expected calibration error over quantile bins of p: the rows' share of each bin times |mean p - alert rate|."""
    if not len(p):
        return math.nan
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    # a bin's share times |mean p - rate| is |its sum of p - its alerts| over all rows
    expected = np.bincount(which, weights=p, minlength=len(edges) - 1)
    observed = np.bincount(which, weights=y.astype(float), minlength=len(edges) - 1)
    return float(np.sum(np.abs(expected - observed)) / len(p))


def _share(part: float, whole: float) -> float:
    return part / whole if whole else math.nan
