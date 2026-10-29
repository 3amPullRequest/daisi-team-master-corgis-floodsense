"""The gated LightGBM pair: a gate screens out quiet area-ticks, and a rater gives the ticks it passes a calibrated
probability of a PUB alert in the area within the hour, which two cut-offs turn into Low/Medium/High.

Both models learn from gold.model_rows only: area-ticks whose label window and feature window are both recorded. So a
missing interval trains neither. The gate learns from every recorded tick, with or without an alert after it; the rater
learns from the recorded ticks the gate passes.

One target, a PUB alert of either kind within the hour (gold.area_labels.alert_label, as scoreboard.tick_labels counts
it): the 90% warnings are what Medium is for, and flash floods are too few to learn alone (36 on 20 days in training,
21 Oct). But a flash flood is what the Round 1 targets are about, so a flood event flagged before PUB counts
`flood_value` times another event (scoreboard.FLOOD_EVENT_VALUE, 10) when the Medium cut-off is chosen, as when
train.py compares fits. The rows a flood followed can also weigh more in training (`flood_weight`, in both models, the
gate's recall and the area prior), but by default they don't: on 22 Oct a weight of 10 cost 12 points of PUB events
flagged and caught no more floods (DEVELOPER_NOTES, Results). High is a higher cut-off on the same probability, where
enough High warnings are followed by a flood.

The area is a prior, not a category (`area_effect`, credibility.py): each area's alert rate by Bühlmann-Straub
credibility, the offset (LightGBM's init_score) both models start from. Read as a category, an area that had never
alerted was pushed towards a zero chance whatever the rain, and its quiet rows pulled the rain effects down in every
other area. The prior trusts an area's own history only as far as its exposure and the real spread between areas
warrant, and the trees learn the weather on top of it. `area_effect="category"` fits the pair as before 21 Oct.

Fitting, on the development period of split.make_split (blocked day-folds, an embargo either side of each block):

1. Gate: LightGBM on every development row, with the quiet rows subsampled and weighted back up and the flood rows
   weighted up, starting from each row's area prior. Its out-of-fold scores set its threshold: the highest that still
   passes `gate_recall` of the alert rows' weight. A third of those had no rain within 5 km yet (gold, 15 Oct): the
   early ticks of a storm, which only the forecast and the hour tell from a quiet afternoon. So the gate reads every
   input the rater does.
2. Rater: LightGBM on the rows the out-of-fold gate passed, weighted and started the same way, risk rising with rain
   (near the area and around it) and with the forecast.
3. Calibration: centered isotonic regression of the label on the rater's pooled out-of-fold scores. Rows the gate
   screens out get the alert rate of the screened-out rows.
4. Cut-offs, on the pooled out-of-fold probabilities, judged by scoreboard.score over the development period as
   baseline.fit judges the rule: Medium flags the most PUB events before PUB within the false-alarm budget, a flood
   event counting `flood_value` times (`weighted_flagged`); High is the lowest higher cut-off whose High warnings are
   followed by a flash flood often enough. Or both are fixed chances (`medium_chance`, `high_chance`): v11 (24 Oct)
   and final_1 (25 Oct, v12's design) have Medium from 1% and High from 5%, and final_2 Medium from 3%, as no cut-off
   on p_alert brings 1 in 3 High warnings a flood.
5. Both models are refitted on the whole development period, with the rounds the folds chose. Every out-of-fold score
   comes from a prior computed without its own block, so no score reads its own block's alerts through the prior.

The test split is never read here.

Scoring a row, which is what a saved pair does (`GatedPair.p_alert`, `predict`):

- A row missing a required feature isn't rated: its p_alert is NaN and its tier Low. A missing optional feature
  (a weather outage, no upwind in a calm) is passed on as missing, which LightGBM reads as such.
- The gate scores every other row, and those at or above its threshold go on to the rater. Both models' scores start
  from the row's area prior (`area_offsets`; an area the prior hasn't seen gets the island's). The rater's score is
  turned into a probability by the calibration, never below the alert rate of the rows the gate screens out. The rows
  the gate stops get that rate (`quiet_p`).
- `tiers` turns the probability into Low, Medium or High with the two cut-offs, and `reasons` gives each warned row a
  plain-English reason for the app: the two inputs that raised the rater's score most (its SHAP contributions, and the
  area prior beside them), in words a resident can act on, or the weather at hand when no single input stands out.
- `save` and `load` keep the pair as two LightGBM model files and a JSON of everything else, and refuse to load a pair
  saved for different inputs. A pair saved before 21 Oct (the area as a category, no offsets) loads and scores as it
  did.

Some choices that shape the models: both are small and heavily regularised, because the alert days to learn from are few
(about 160 in training); the trees are constrained so that the chance of an alert can only rise with the rain, near the
area and around it, the forecast class and the tide (`RISING`); the quiet rows are subsampled and weighted back up
(`_quiet_sample`), so ten million rows train quickly without biasing the totals; and every cut-off must lie above the
probability of the screened-out rows, so that a cut-off cannot become "warn everywhere".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import contract as C
from .credibility import area_days, area_offsets, area_prior
from .scoreboard import FLOOD_EVENT_VALUE, TARGETS, score
from .split import Split, _sgt

AREA, HOUR = "area", "hour_sgt"
INPUTS = (
    AREA,
    HOUR,
    *C.MODEL_FEATURES,
)  # what both models can read, in this order (Settings.inputs leaves some out)
# how the pair reads the area: a Bühlmann-Straub prior as the models' offset, or (before 21 Oct) a category input
CREDIBILITY, CATEGORY = "credibility", "category"
AREA_EFFECTS = (CREDIBILITY, CATEGORY)
RAIN = tuple(C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES)
NEARBY_RAIN = ("rain_nearby_30min_mm", "rain_upwind_30min_mm")
TIDE = ("tide_height_m", "minutes_to_high_tide")
HIGH_TIDE_M = 2.5  # above Chart Datum; Tanjong Pagar's high waters run about 2.2-3.2 m (NEA, Oct-Nov 2026)
# the alert chance may only rise with these, the drains' response among them (more water in the drains, or a new high
# in the storm, never lowers it); the storm so far (contract.STORM_HISTORY) may move it either way, as the prior rain may
RISING = {
    *RAIN, "forecast_rain_class", *C.FORECAST_GROUPED, *NEARBY_RAIN, *C.PUB_ALERTS, "tide_height_m", *C.DRAIN_RESPONSE
}

# Few events (about 160 alert days in training), so both models are small and heavily regularised.
GATE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
    "lambda_l2": 1.0,
    "cat_smooth": 20.0,
    "min_data_per_group": 200,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}
RATER_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 100,
    "lambda_l2": 5.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "cat_smooth": 20.0,
    "min_data_per_group": 100,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(frozen=True)
class Settings:
    """gate_recall: the share of the alert rows' weight that the gate must pass, out of fold. The quiet shares: how many
    quiet rows each model trains on, weighted back up. flood_weight: what a row a flash flood followed weighs against
    one only a warning followed, in training, in the gate's recall and in the area prior's claims. flood_value: what a
    flood event flagged before PUB counts when the Medium cut-off is chosen. area_effect: CREDIBILITY (the area prior as
    both models' offset) or CATEGORY (the area as an input, as before 21 Oct). patience: rounds without a better mean
    out-of-fold loss before the folds stop. max_false_alarms is per area-month, as in baseline.fit. medium_chance and
    high_chance fix the tiers' cut-offs on p_alert (the tiers by chance, 24 Oct: Medium from 1%, High from 5%) instead
    of choosing Medium within the false-alarm budget and High by its flood precision; None chooses them so."""

    gate_recall: float = 0.9
    gate_quiet_share: float = 0.05
    rater_quiet_share: float = 0.25
    flood_weight: float = 1.0
    flood_value: float = FLOOD_EVENT_VALUE
    area_effect: str = CREDIBILITY
    # the credibility prior's recency half-life in days (credibility.recency); None weighs every day alike, as v5
    prior_half_life_days: float | None = None
    max_rounds: int = 2000
    patience: int = 100
    max_false_alarms: float = 1.0
    min_high_precision: float = TARGETS["high_precision"]
    cut_off_candidates: int = 60
    medium_chance: float | None = None
    high_chance: float | None = None
    seed: int = 0
    # model features left out: those outside the feature set (contract.BASE_FEATURES by default: v5's), and to measure
    # what a group adds (train.py fits the pair with and without each group in contract.FEATURE_GROUPS that
    # --compare-without names)
    drop: tuple[str, ...] = tuple(c for c in C.MODEL_FEATURES if c not in C.BASE_FEATURES)

    def __post_init__(self):
        if self.area_effect not in AREA_EFFECTS:
            raise ValueError(
                f"area_effect is one of {AREA_EFFECTS}, not {self.area_effect!r}"
            )
        if self.flood_weight < 1 or self.flood_value < 1:
            raise ValueError(
                "flood_weight and flood_value are at least 1: a flood weighs no less than a warning"
            )
        if self.prior_half_life_days is not None and not self.prior_half_life_days > 0:
            raise ValueError(f"prior_half_life_days is positive, not {self.prior_half_life_days!r}")
        for name in ("medium_chance", "high_chance"):
            chance = getattr(self, name)
            if chance is not None and not 0 < chance < 1:
                raise ValueError(f"{name} is a chance between 0 and 1, not {chance!r}")
        if None not in (self.medium_chance, self.high_chance) and not self.medium_chance < self.high_chance:
            raise ValueError(
                f"high_chance ({self.high_chance}) must be above medium_chance ({self.medium_chance})"
            )

    @property
    def inputs(self) -> tuple[str, ...]:
        return inputs_without(self.drop, area=self.area_effect == CATEGORY)


DEFAULTS = Settings()


def inputs_without(drop: tuple[str, ...] = (), *, area: bool = True) -> tuple[str, ...]:
    """INPUTS without the model features in `drop`, and without the area unless `area` (it is an input only when the
    pair reads it as a category)."""
    unknown = set(drop) - set(C.MODEL_FEATURES)
    if unknown:
        raise ValueError(f"not model features: {sorted(unknown)}")
    return tuple(c for c in INPUTS if c not in drop and (area or c != AREA))


def design(
    frame: pd.DataFrame, areas: tuple[str, ...], inputs: tuple[str, ...] = INPUTS
) -> pd.DataFrame:
    """The models' inputs from gold rows: the area as a category, the SGT hour of day, and the model features (those
    of `inputs`, in its order)."""
    ts = _sgt(frame["ts"])
    columns = {
        AREA: pd.Categorical(frame[AREA], categories=list(areas)),
        HOUR: (ts.dt.hour + ts.dt.minute / 60).astype("float32").to_numpy(),
        **{
            c: pd.to_numeric(frame[c]).astype("float32").to_numpy()
            for c in C.MODEL_FEATURES
            if c in inputs
        },
    }
    return pd.DataFrame(columns, index=frame.index)[list(inputs)]


@dataclass
class GatedPair:
    """The fitted pair: gate, threshold, rater, calibration and cut-offs. `predict` writes ml.predictions rows.

    `area_offsets` is the area prior both models start from (log relative risk per area), and `gate_base` and
    `rater_base` the log-odds they start from besides; a pair that reads the area as a category has none, and its trees
    carry their own start, as LightGBM fits them without an init_score."""

    areas: tuple[str, ...]
    gate: lgb.Booster
    gate_threshold: float
    rater: lgb.Booster
    calibration_x: np.ndarray
    calibration_y: np.ndarray
    quiet_p: float
    medium_p: float
    high_p: float = math.inf
    inputs: tuple[str, ...] = INPUTS
    area_offsets: dict[str, float] | None = None
    gate_base: float = 0.0
    rater_base: float = 0.0
    flood_weight: float = 1.0
    # the rainfall rule as a safety override (ordinal.add_override, 23 Oct): (window in minutes, mm), or none
    override: tuple[int, float] | None = None

    @property
    def features(self) -> list[str]:
        return [c for c in self.inputs if c in C.MODEL_FEATURES]

    @property
    def name(self) -> str:
        over = f" R>={self.override[1]:g}mm/{self.override[0]}min" if self.override else ""
        return f"gated_lightgbm M>={self.medium_p:.3g} H>={self.high_p:.3g}{over}"

    def offsets(self, frame: pd.DataFrame) -> np.ndarray:
        """Each row's area offset (0 everywhere for a pair without a prior)."""
        if self.area_offsets is None:
            return np.zeros(len(frame))
        return area_offsets(frame[AREA], self.area_offsets)

    def gate_p(self, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        return _scored(self.gate, X, self.gate_base, offset, self.area_offsets)

    def rater_p(self, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        return _scored(self.rater, X, self.rater_base, offset, self.area_offsets)

    def p_alert(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(passed, p_alert) per row. A row missing a required feature the pair reads isn't rated: p_alert is NaN, and
        its tier Low. A missing optional feature (a weather outage, no upwind in a calm) is read as missing."""
        X = design(frame, self.areas, self.inputs)
        offset = self.offsets(frame)
        required = [c for c in C.REQUIRED_FEATURES if c in self.features]
        known = X[required].notna().all(axis=1).to_numpy()
        passed = np.zeros(len(X), dtype=bool)
        if known.any():
            passed[known] = self.gate_p(X[known], offset[known]) >= self.gate_threshold
        p = np.where(known, self.quiet_p, np.nan)
        if passed.any():
            p[passed] = self.calibrate(self.rater_p(X[passed], offset[passed]))
        return passed, p

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return _calibrated(raw, self.calibration_x, self.calibration_y, self.quiet_p)

    def tiers(self, p: np.ndarray) -> np.ndarray:
        return _tiers(p, self.medium_p, self.high_p)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rows for ml.predictions, with the reason the app shows for every warning. Where the override is set, a rated
        row whose rain passes it is at least Medium."""
        _, p = self.p_alert(frame)
        tier = self.tiers(p)
        forced = override_mask(frame, self.override) & np.isfinite(p)
        tier = np.where(forced & (tier == "Low"), "Medium", tier)
        reason = self.reasons(frame, tier)
        only_rule = forced & (p < self.medium_p)
        if only_rule.any():
            reason[only_rule] = [override_reason(r, self.override) for r in frame[only_rule].to_dict("records")]
        return pd.DataFrame(
            {
                "model": self.name,
                "area": frame[AREA],
                "ts": frame["ts"],
                "tier": tier,
                "p_alert": p,
                "reason": reason,
            }
        )

    def reasons(self, frame: pd.DataFrame, tier: np.ndarray) -> pd.Series:
        """Plain English for each warned row: the two inputs that raised the rater's score most (its SHAP values, and
        the area prior beside them when the pair has one)."""
        reason = pd.Series(None, index=frame.index, dtype=object)
        warned = np.asarray(tier) != "Low"
        if not warned.any():
            return reason
        rows = frame[warned]
        contributions = self.rater.predict(
            design(rows, self.areas, self.inputs), pred_contrib=True
        )
        names = list(self.inputs)
        if self.area_offsets is not None and AREA not in self.inputs:
            # the prior shifts the score as an input's contribution would: put it beside them, before the bias
            contributions = np.column_stack(
                [contributions[:, :-1], self.offsets(rows), contributions[:, -1]]
            )
            names.append(AREA)
        texts = []
        for row, contribution in zip(rows.to_dict("records"), contributions):
            ranked = np.argsort(-contribution[:-1])  # the last column is the bias
            texts.append(reason_text([names[i] for i in ranked if contribution[i] > 0], row))
        reason[warned] = texts
        return reason

    def chances(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """p_alert per row (NaN where unrated), as verify.py compares models' chances."""
        _, p = self.p_alert(frame)
        return {"p_alert": p}

    @property
    def area_priors(self) -> dict[str, dict[str, float]]:
        return {} if self.area_offsets is None else {"alert": self.area_offsets}

    def logged_cut_offs(self) -> dict[str, float]:
        """The cut-offs and gate threshold as train.py logs them (pair_*), which verify.py checks the registered pair
        against."""
        out = {
            "pair_medium_p": self.medium_p,
            "pair_quiet_p": self.quiet_p,
            "pair_gate_threshold": self.gate_threshold,
        }
        if math.isfinite(self.high_p):
            out["pair_high_p"] = self.high_p
        if self.override is not None:
            out["pair_override_mm"] = self.override[1]
        return out

    def save(self, directory) -> None:
        """Two LightGBM model files and the rest as JSON, so loading needs only lightgbm and numpy."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.gate.save_model(str(out / "gate.txt"))
        self.rater.save_model(str(out / "rater.txt"))
        settings = {
            "inputs": list(self.inputs),
            "areas": list(self.areas),
            "gate_threshold": self.gate_threshold,
            "calibration_x": [float(x) for x in self.calibration_x],
            "calibration_y": [float(y) for y in self.calibration_y],
            "quiet_p": self.quiet_p,
            "medium_p": self.medium_p,
            "high_p": None if math.isinf(self.high_p) else self.high_p,
            "area_offsets": None
            if self.area_offsets is None
            else {a: float(v) for a, v in self.area_offsets.items()},
            "gate_base": self.gate_base,
            "rater_base": self.rater_base,
            "flood_weight": self.flood_weight,
            "override": None
            if self.override is None
            else {"window_min": self.override[0], "mm": self.override[1]},
        }
        (out / "pair.json").write_text(json.dumps(settings, indent=1))

    @classmethod
    def load(cls, directory) -> GatedPair:
        src = Path(directory)
        s = json.loads((src / "pair.json").read_text())
        inputs = tuple(s["inputs"])
        # a pair saved without some features loads; one that reads a feature this code doesn't know can't
        if inputs != tuple(c for c in INPUTS if c in inputs):
            raise ValueError(
                f"saved for inputs {s['inputs']}, but this code reads {list(INPUTS)}"
            )
        # pairs saved before 21 Oct have no prior and no base: their trees carry their own start
        return cls(
            areas=tuple(s["areas"]),
            gate=lgb.Booster(model_file=str(src / "gate.txt")),
            gate_threshold=s["gate_threshold"],
            rater=lgb.Booster(model_file=str(src / "rater.txt")),
            calibration_x=np.asarray(s["calibration_x"]),
            calibration_y=np.asarray(s["calibration_y"]),
            quiet_p=s["quiet_p"],
            medium_p=s["medium_p"],
            high_p=math.inf if s["high_p"] is None else s["high_p"],
            inputs=inputs,
            area_offsets=s.get("area_offsets"),
            gate_base=s.get("gate_base", 0.0),
            rater_base=s.get("rater_base", 0.0),
            flood_weight=s.get("flood_weight", 1.0),
            # pairs saved before 23 Oct have no override
            override=None
            if s.get("override") is None
            else (int(s["override"]["window_min"]), float(s["override"]["mm"])),
        )


@dataclass
class Fit:
    """What fitting learned about the pair, out of fold, for logging. One entry per development row in `rows` order."""

    rows: pd.DataFrame  # area, ts, alert_label, flood_label
    gate_score: np.ndarray
    passed: np.ndarray
    p_alert: np.ndarray
    tier: np.ndarray
    curve: pd.DataFrame  # every Medium candidate's out-of-fold scores
    stats: dict[str, float] = field(default_factory=dict)
    importance: pd.DataFrame | None = None
    area_prior: pd.DataFrame | None = None  # the final models' prior (credibility.AreaPrior.table), if they have one

    def predictions(self, *, warned_only: bool = True) -> pd.DataFrame:
        """Out-of-fold (area, ts, tier, p_alert) rows, as scoreboard.score takes them."""
        out = self.rows[[AREA, "ts"]].assign(tier=self.tier, p_alert=self.p_alert)
        return out[self.tier != "Low"] if warned_only else out

    def reliability(self, bins: int = 10) -> pd.DataFrame:
        """Calibration of the passed rows: mean p_alert against the alert rate, per decile of p_alert."""
        p = self.p_alert[self.passed]
        y = self.rows["alert_label"].to_numpy()[self.passed]
        if len(np.unique(p)) < 2:
            return pd.DataFrame(
                {
                    "p_alert_range": [f"{p.min():.4g}" if len(p) else ""],
                    "rows": [len(p)],
                    "mean_p_alert": [p.mean() if len(p) else np.nan],
                    "alert_rate": [y.mean() if len(y) else np.nan],
                }
            )
        decile = pd.qcut(p, q=bins, duplicates="drop")
        table = pd.DataFrame({"p": p, "y": y}).groupby(decile, observed=True)
        out = table.agg(
            rows=("p", "size"), mean_p_alert=("p", "mean"), alert_rate=("y", "mean")
        )
        return out.reset_index(names="p_alert_range").astype({"p_alert_range": str})


def fit(
    rows: pd.DataFrame,
    alerts: pd.DataFrame,
    split: Split,
    *,
    missing=None,
    blind=None,
    settings: Settings = DEFAULTS,
) -> tuple[GatedPair, Fit]:
    """Fit the pair on the development period of `split` (module docstring).

    `rows` is gold.model_rows (split 'train'); rows outside the development period are left out. `alerts` is
    silver.area_alerts. `missing` and `blind` list the ticks whose labels or features are unknown (gold.label_ticks,
    gold.feature_ticks), for scoreboard.score.
    """
    # only fitting needs scikit-learn: loading a saved pair (serving.py) must need no more than train.log_model pins
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score

    rows = development_rows(rows, split)
    areas = tuple(sorted(rows[AREA].unique()))
    X = design(rows, areas, settings.inputs)
    y = rows["alert_label"].to_numpy(dtype=np.int8)
    flood = rows["flood_label"].to_numpy(dtype=np.int8)
    ts = rows["ts"]
    if not y.any():
        raise ValueError("no alerts in the development rows")
    lift = np.where(flood == 1, settings.flood_weight, 1.0)  # a flood row's weight
    fold_offsets, prior = _priors(rows, split, settings)
    offsets = None if prior is None else prior.offsets

    gate_weight = _quiet_weight(y, settings.gate_quiet_share, settings.seed) * lift
    gate_score, gate_rounds = _out_of_fold(
        X, y, gate_weight, ts, rows[AREA], fold_offsets, split, GATE_PARAMS, settings
    )
    threshold = recall_threshold(gate_score, y, settings.gate_recall, weight=lift)
    passed = gate_score >= threshold

    rater_weight = (
        _quiet_weight(y[passed], settings.rater_quiet_share, settings.seed)
        * lift[passed]
    )
    raw, rater_rounds = _out_of_fold(
        X[passed],
        y[passed],
        rater_weight,
        ts[passed],
        rows[AREA][passed],
        fold_offsets,
        split,
        RATER_PARAMS,
        settings,
    )
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        raw, y[passed]
    )
    calibration_x, calibration_y = _centered(
        np.asarray(isotonic.X_thresholds_, dtype=float),
        np.asarray(isotonic.y_thresholds_, dtype=float),
    )
    quiet_p = float(y[~passed].mean()) if (~passed).any() else 0.0
    p = np.full(len(rows), quiet_p)
    p[passed] = _calibrated(raw, calibration_x, calibration_y, quiet_p)

    medium_p, high_p, curve = cut_offs(
        rows, p, passed, alerts, split, missing=missing, blind=blind, settings=settings
    )
    gate, gate_base = _train(
        X, y, gate_weight, rows[AREA], offsets, GATE_PARAMS, gate_rounds, settings
    )
    rater, rater_base = _train(
        X[passed],
        y[passed],
        rater_weight,
        rows[AREA][passed],
        offsets,
        RATER_PARAMS,
        rater_rounds,
        settings,
    )
    pair = GatedPair(
        areas=areas,
        gate=gate,
        gate_threshold=float(threshold),
        rater=rater,
        calibration_x=calibration_x,
        calibration_y=calibration_y,
        quiet_p=quiet_p,
        medium_p=medium_p,
        high_p=high_p,
        inputs=settings.inputs,
        area_offsets=offsets,
        gate_base=gate_base,
        rater_base=rater_base,
        flood_weight=settings.flood_weight,
    )

    tier = pair.tiers(p)
    gate_only = score(
        rows.loc[passed, [AREA, "ts"]].assign(tier="Medium"),
        alerts,
        start=split.start,
        end=split.dev_end,
        missing=missing,
        blind=blind,
    ).metrics(ci=False)
    stats = {
        "rows": len(rows),
        "alert_rows": int(y.sum()),
        "flood_rows": int(flood.sum()),
        "flood_weight": settings.flood_weight,
        "flood_value": settings.flood_value,
        "area_prior": float(prior is not None),
        "gate_rounds": gate_rounds,
        "gate_threshold": float(threshold),
        "gate_pass_share": float(passed.mean()),
        "gate_alert_row_recall": float(passed[y == 1].mean()),
        # what the threshold holds to gate_recall: the alert rows passed, each weighing its lift
        "gate_weighted_recall": float(
            np.sum(lift[y == 1] * passed[y == 1]) / np.sum(lift[y == 1])
        ),
        "gate_flood_row_recall": float(passed[flood == 1].mean())
        if flood.any()
        else math.nan,
        # PUB events the gate leaves the rater a chance to flag before PUB: a passed tick in force at PUB's alert
        "gate_alert_event_recall": gate_only["alert_flagged_before_pub"],
        "gate_flood_event_recall": gate_only["flood_flagged_before_pub"],
        "rater_rows": int(passed.sum()),
        "rater_rounds": rater_rounds,
        "quiet_p": quiet_p,
        "medium_p": medium_p,
        "high_p": high_p,
        "brier": float(np.mean((p - y) ** 2)),
        "brier_base_rate": float(np.mean((y.mean() - y) ** 2)),
        # as the ordinal model reports them: the rows warned, and average precision against each label's base rate
        # (the pair's one chance stands for the flood chance too)
        "warned_row_share": float((tier != "Low").mean()),
        "alert_row_recall": float((tier != "Low")[y == 1].mean()),
        "flood_row_recall": float((tier != "Low")[flood == 1].mean()) if flood.any() else math.nan,
        "alert_pr_auc": float(average_precision_score(y, p)),
        "flood_pr_auc": float(average_precision_score(flood, p)) if flood.any() else math.nan,
        "alert_base_rate": float(y.mean()),
        "flood_base_rate": float(flood.mean()),
    }
    if prior is not None:
        stats |= {
            # the exposure (recorded days) at which an area's own history gets half the weight
            "area_prior_k_days": prior.k,
            "area_prior_credibility_mean": float(prior.table["credibility"].mean()),
            "area_prior_offset_min": float(prior.table["offset"].min()),
            "area_prior_offset_max": float(prior.table["offset"].max()),
        }
    importance = pd.DataFrame(
        {
            "input": settings.inputs,
            "gate_gain": pair.gate.feature_importance("gain"),
            "rater_gain": pair.rater.feature_importance("gain"),
        }
    ).sort_values("rater_gain", ascending=False)
    report = Fit(
        rows=rows[[AREA, "ts", "alert_label", "flood_label"]],
        gate_score=gate_score,
        passed=passed,
        p_alert=p,
        tier=tier,
        curve=curve,
        stats=stats,
        importance=importance,
        area_prior=None if prior is None else prior.table,
    )
    return pair, report


def development_rows(rows: pd.DataFrame, split: Split) -> pd.DataFrame:
    """The rows inside the development period, in a fixed order (so subsamples repeat). Refuses a row whose label or
    features aren't known: gold.model_rows holds none, and a model must never learn from one. Rows that already are
    that (this function's own output) come back as they are, not copied: ten million rows are 1.4 GB, and a sweep's
    serverless driver ran out of memory on 23 Oct copying them for every fit."""
    check_recorded(rows)
    inside = split.dev(rows["ts"])
    if inside.all() and _in_order(rows):
        return rows
    return rows[inside].sort_values(["ts", AREA], kind="stable").reset_index(drop=True)


def _in_order(rows: pd.DataFrame) -> bool:
    """Whether rows are sorted by (ts, area) on a fresh RangeIndex, as development_rows leaves them."""
    index = rows.index
    if not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1):
        return False
    step = np.diff(_sgt(rows["ts"]).to_numpy())
    if (step < np.timedelta64(0)).any():
        return False
    area = rows[AREA]
    codes = (
        area.cat.codes.to_numpy()
        if isinstance(area.dtype, pd.CategoricalDtype) and area.cat.categories.is_monotonic_increasing
        else pd.factorize(area.astype(object), sort=True)[0]
    )
    return not (np.diff(codes)[step == np.timedelta64(0)] < 0).any()


def check_recorded(rows: pd.DataFrame) -> None:
    """Raise if a row has a missing label or required feature: it would come from a missing interval."""
    unknown = (
        rows[["alert_label", "flood_label", *C.REQUIRED_FEATURES]].isna().any(axis=1)
    )
    if unknown.any():
        first = rows.loc[unknown, [AREA, "ts"]].iloc[0].to_dict()
        raise ValueError(
            f"{int(unknown.sum())} rows have a missing label or feature (first: {first}); models learn only from "
            "gold.model_rows, where both windows are recorded"
        )


def usable_alerts(alerts: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The alerts the rows can learn from: the tick before each is a row of its area (an alert row, as the tick's
    label window holds the alert). make_split balances its blocks on these, so no block is made of days without rows."""
    positive = rows[rows["alert_label"] == 1]
    keys = set(zip(positive[AREA], _sgt(positive["ts"])))
    tick = (_sgt(alerts["alert_ts"]) - pd.Timedelta(1, "ns")).dt.floor(
        f"{C.TICK_MINUTES}min"
    )
    keep = [(a, t) in keys for a, t in zip(alerts[AREA], tick)]
    return alerts[np.asarray(keep, dtype=bool)]


def recall_threshold(
    scores: np.ndarray, y: np.ndarray, recall: float, weight: np.ndarray | None = None
) -> float:
    """The highest threshold that passes (score >= threshold) at least `recall` of the rows with label 1, each row
    counting its `weight` (a flood row weighs flood_weight; all 1 when None)."""
    positive = y == 1
    if not positive.any():
        raise ValueError("no alert rows to set the gate on")
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    order = np.argsort(scores[positive], kind="stable")
    s, w = scores[positive][order], w[positive][order]
    # the lowest rows may be dropped while their weight stays within (1 - recall) of the total
    dropped = int(np.sum(np.cumsum(w) <= (1 - recall) * w.sum() + 1e-9))
    return float(s[min(dropped, len(s) - 1)])


def weighted_flagged(metrics: dict[str, float], flood_weight: float) -> float:
    """The share of PUB events flagged before PUB, a flood event counting `flood_weight` times: once as the PUB event it
    belongs to and flood_weight - 1 times more as a flood. Without flood events, the plain alert share."""
    alerts, floods = metrics.get("alert_events", 0), metrics.get("flood_events", 0)
    a, f = metrics["alert_flagged_before_pub"], metrics.get("flood_flagged_before_pub")
    extra = (flood_weight - 1.0) * floods
    if not floods or not extra or f is None or math.isnan(f) or not alerts:
        return a
    return (alerts * a + extra * f) / (alerts + extra)


def cut_offs(
    rows: pd.DataFrame,
    p: np.ndarray,
    passed: np.ndarray,
    alerts: pd.DataFrame,
    split: Split,
    *,
    missing=None,
    blind=None,
    settings: Settings = DEFAULTS,
) -> tuple[float, float, pd.DataFrame]:
    """Medium and High cut-offs on out-of-fold probabilities `p` (one per row), as baseline.fit sets the rule's.

    Medium: the cut-off that flags the most PUB alert events before PUB within `max_false_alarms` per area-month, a
    flood event counting `flood_value` times (`weighted_flagged`); ties go to the longer mean lead, then to fewer
    warned hours. High: the lowest higher cut-off whose High warnings are followed by a flash flood at least
    `min_high_precision` of the time, else never. `settings.medium_chance` or `high_chance` fixes that cut-off instead
    (the tiers by chance); the curve is computed either way. Returns both and the Medium curve.

    Every cut-off is above the probability of the rows the gate screens out: calibration floors the passed rows at it,
    and a cut-off there would warn every row, which the scoreboard can't fault (one endless warning per area holds
    every alert, so it is never a false alarm).
    """
    screened_out = p[~passed].max() if (~passed).any() else -math.inf
    candidates = [
        c
        for c in _candidates(p[passed], settings.cut_off_candidates)
        if c > screened_out
    ]
    frame = rows[[AREA, "ts"]]

    def metrics(medium: float, high: float = math.inf) -> dict[str, float]:
        warned = p >= medium
        preds = frame[warned].assign(tier=_tiers(p[warned], medium, high))
        board = score(
            preds,
            alerts,
            start=split.start,
            end=split.dev_end,
            missing=missing,
            blind=blind,
        )
        return board.metrics(ci=False)

    curve = []
    for c in candidates:
        m = metrics(c)
        curve.append(
            {
                "medium_p": c,
                "flagged": m["alert_flagged_before_pub"],
                "lead_min": m["alert_mean_lead_min"],
                "false_alarms": m["false_alarms_per_area_month"],
                "warned_hours": m["warned_hours_per_area_month"],
                "flood_flagged": m["flood_flagged_before_pub"],
                "weighted_flagged": weighted_flagged(m, settings.flood_value),
                "row_share": float((p >= c).mean()),
            }
        )
    curve = pd.DataFrame(curve)
    if settings.medium_chance is not None:
        medium = settings.medium_chance
        if medium <= screened_out:
            raise ValueError(
                f"a Medium cut-off of {medium:g} would warn the rows the gate screens out (their chance {screened_out:.3g})"
            )
    else:
        within = curve[curve["false_alarms"] <= settings.max_false_alarms]
        if within.empty:
            raise ValueError(
                f"every cut-off raises more than {settings.max_false_alarms} false alarms per area-month"
            )
        best = within.sort_values(
            ["weighted_flagged", "lead_min", "warned_hours"], ascending=[False, False, True]
        ).iloc[0]
        medium = float(best["medium_p"])
    if settings.high_chance is not None:
        if settings.high_chance <= medium:
            raise ValueError(f"High from {settings.high_chance:g} would sit at or below Medium ({medium:.3g})")
        return medium, settings.high_chance, curve

    high = math.inf
    for c in (c for c in candidates if c > medium):
        m = metrics(medium, c)
        if m["high_warnings"] and m["high_precision"] >= settings.min_high_precision:
            high = float(c)
            break
    return medium, high, curve


def _candidates(p: np.ndarray, n: int) -> list[float]:
    """Up to n cut-offs: the probabilities that warn on a geometric range of shares of the passed rows, from 1 in
    100,000 to all of them, so most fall in the tail where useful cut-offs lie (out of fold, 15 Oct: 0.1%)."""
    if not len(p):
        return []
    shares = np.geomspace(1e-5, 1, n)
    return [float(v) for v in np.unique(np.quantile(p, 1 - shares))]


def _tiers(p: np.ndarray, medium: float, high: float) -> np.ndarray:
    return np.select([p >= high, p >= medium], ["High", "Medium"], "Low")


def _centered(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered isotonic regression: each flat block of an isotonic fit (x, y) becomes one point at its middle, joined
    linearly to the next. The calibration is kept on average, but distinct scores get distinct probabilities, so a
    cut-off can fall inside what was one block: one block held thousands of storm ticks on 15 Oct, and the
    out-of-fold curve jumped from 45% of events flagged to 22% across it."""
    if len(y) < 2:
        return x, y
    starts = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
    ends = np.r_[starts[1:], len(y)] - 1
    return (x[starts] + x[ends]) / 2, y[starts]


def _calibrated(
    raw: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float
) -> np.ndarray:
    """Raw rater scores as probabilities: the centered isotonic fit (x, y), flat beyond its ends, never below the
    alert rate of the rows the gate screens out."""
    return np.maximum(np.interp(raw, x, y), floor)


def _quiet_sample(
    y: np.ndarray, share: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(keep, weight): every alert row and `share` of the quiet ones, weighted 1/share so totals stay unbiased."""
    keep = (y == 1) | (np.random.default_rng(seed).random(len(y)) < share)
    weight = np.where(y == 1, 1.0, 1.0 / share)
    return keep, weight


def _quiet_weight(y: np.ndarray, share: float, seed: int) -> np.ndarray:
    """Each row's training weight from the quiet subsample (`_quiet_sample`), 0 for a row left out of it."""
    keep, weight = _quiet_sample(y, share, seed)
    return np.where(keep, weight, 0.0)


def _params(base: dict, settings: Settings) -> dict:
    return {
        **base,
        "seed": settings.seed,
        "monotone_constraints": [1 if c in RISING else 0 for c in settings.inputs],
    }


def _priors(
    rows: pd.DataFrame, split: Split, settings: Settings
) -> tuple[list[dict[str, float] | None], object | None]:
    """The area offsets of each fold's models, from that fold's training days only, and the prior of the final models,
    from the whole development period (credibility.area_prior); Nones when the pair reads the area as a category. With
    a recency half-life, a fold's prior weighs both ways from the middle of its block, the final one back from its last
    day (credibility.py)."""
    if settings.area_effect != CREDIBILITY:
        return [None] * len(split.folds), None
    days = area_days(rows)
    day = pd.Series(days["day"])
    weight, half_life = settings.flood_weight, settings.prior_half_life_days
    folds = [
        area_prior(
            days[fold.train(day)],
            flood_weight=weight,
            half_life_days=half_life,
            reference=fold.valid_start + (fold.valid_end - fold.valid_start) / 2,
            two_sided=True,
        ).offsets
        for fold in split.folds
    ]
    return folds, area_prior(days, flood_weight=weight, half_life_days=half_life)


def _offsets(area: pd.Series, offsets: dict[str, float] | None) -> np.ndarray:
    return np.zeros(len(area)) if offsets is None else area_offsets(area, offsets)


def _init_base(y: np.ndarray, weight: np.ndarray, offset: np.ndarray) -> float:
    """The log-odds both models start from besides the area offsets, so that the start matches the weighted alert rate
    on average. LightGBM doesn't boost from the average when given an init_score (checked, lightgbm 4.7)."""
    rate = float(np.average(y, weights=weight))
    return math.log(rate / (1 - rate)) - math.log(
        float(np.average(np.exp(offset), weights=weight))
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _scored(
    booster: lgb.Booster,
    X: pd.DataFrame,
    base: float,
    offset: np.ndarray,
    offsets: dict | None,
    rounds: int | None = None,
) -> np.ndarray:
    """A model's probabilities from its trees, its base and the rows' area offsets. A pair without a prior scores as
    LightGBM does, its trees holding their own start."""
    if offsets is None and base == 0.0:
        return booster.predict(X, num_iteration=rounds)
    return _sigmoid(booster.predict(X, raw_score=True, num_iteration=rounds) + base + offset)


def _out_of_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    ts: pd.Series,
    area: pd.Series,
    fold_offsets: list[dict[str, float] | None],
    split: Split,
    params: dict,
    settings: Settings,
) -> tuple[np.ndarray, int]:
    """Out-of-fold probabilities for every row, and the rounds the folds agree on (the best mean validation loss).

    `weight` is each row's training weight, 0 for the quiet rows left out of the subsample. Each fold trains on its
    training rows of the subsample, starting from its own area offsets (`fold_offsets`, computed without its block), and
    scores every row of its block, so the scores cover all rows. The folds grow in step and stop together, as lgb.cv's
    do, which can't give a row a different init_score in each fold.
    """
    keep = weight > 0
    kept = np.flatnonzero(keep)
    p = _params(params, settings)
    data = lgb.Dataset(
        X[keep], label=y[keep], weight=weight[keep], params=p, free_raw_data=False
    ).construct()
    boosters, bases = [], []
    for fold, offsets in zip(split.folds, fold_offsets):
        offset = _offsets(area, offsets)[kept]
        train = np.flatnonzero(fold.train(ts)[kept])
        valid = np.flatnonzero(fold.valid(ts)[kept])
        # without a prior the offsets are 0 and this is LightGBM's boost_from_average, where lgb.cv's folds started
        base = _init_base(y[kept][train], weight[kept][train], offset[train])
        sets = []
        for idx in (train, valid):
            part = data.subset(idx).construct()
            part.set_init_score(base + offset[idx])
            sets.append(part)
        booster = lgb.Booster(params=p, train_set=sets[0])
        booster.add_valid(sets[1], "valid")
        boosters.append(booster)
        bases.append(base)

    rounds, best = 0, math.inf
    for r in range(1, settings.max_rounds + 1):
        losses = []
        for booster in boosters:
            booster.update()
            losses.append(booster.eval_valid()[0][2])
        loss = float(np.mean(losses))
        if loss < best:
            rounds, best = r, loss
        elif r - rounds >= settings.patience:
            break

    scores = np.full(len(y), np.nan)
    for fold, booster, base, offsets in zip(split.folds, boosters, bases, fold_offsets):
        block = fold.valid(ts)
        if block.any():
            raw = booster.predict(X[block], raw_score=True, num_iteration=rounds)
            scores[block] = _sigmoid(raw + base + _offsets(area[block], offsets))
    if np.isnan(scores).any():
        raise ValueError("some development rows fell in no fold's block")
    return scores, rounds


def _train(
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    area: pd.Series,
    offsets: dict[str, float] | None,
    params: dict,
    rounds: int,
    settings: Settings,
) -> tuple[lgb.Booster, float]:
    """A model on every row of the subsample (weight > 0), with the rounds the folds chose, and the base it starts from
    besides the area offsets (`_init_base`)."""
    keep = weight > 0
    offset = _offsets(area, offsets)[keep]
    base = _init_base(y[keep], weight[keep], offset)
    p = _params(params, settings)
    data = lgb.Dataset(
        X[keep], label=y[keep], weight=weight[keep], init_score=base + offset, params=p
    )
    return lgb.train(p, data, num_boost_round=rounds), base


def override_mask(frame: pd.DataFrame, override: tuple[int, float] | None) -> np.ndarray:
    """Where the rainfall rule's override (window in minutes, mm) warns: that window's rain at or above its
    threshold. None warns nowhere."""
    if override is None:
        return np.zeros(len(frame), dtype=bool)
    window, mm = override
    return (pd.to_numeric(frame[C.rain_column(window)]) >= mm).fillna(False).to_numpy(dtype=bool)


def override_reason(row: dict, override: tuple[int, float]) -> str:
    """The rule's words, for a warning only the override raised."""
    window, _ = override
    return f"{row[C.rain_column(window)]:.0f} mm of rain within {C.GAUGE_RADIUS_KM:g} km in the last {window} min"


def reason_text(raised: list[str], row: dict) -> str:
    """A warned row's reason from the inputs that raised its score, most first: the first two that make words, or the
    weather at hand when none does (inputs can act only together). The tide only adds to a reason: a high tide alone
    never floods a drain."""
    # the drains' two rises say the same in words, so a phrase counts once
    phrases = list(dict.fromkeys(p for p in (_phrase(c, row) for c in raised) if p))[:2]
    weather = next(
        (
            w
            for w in (_phrase(c, row) for c in (*RAIN, *NEARBY_RAIN, "forecast_rain_class"))
            if w
        ),
        "conditions like those before past PUB alerts",
    )
    tide_only = all(c in TIDE for c in raised if _phrase(c, row))
    if not phrases:
        phrases = [weather]
    elif tide_only:
        phrases = [weather, phrases[0]]
    return "; ".join(phrases)


def _phrase(name: str, row: dict) -> str | None:
    """The app's words for one input of a warned row, or None for inputs that don't make a reason a resident can act
    on (the hour, gauge counts, forecast age, the dew point), or that didn't add to the risk here (a low tide)."""
    value = row.get(name)
    if name in RAIN and value is not None and value >= 1:
        minutes = name.removeprefix("rain_").removesuffix("min_mm")
        return f"{value:.0f} mm of rain within {C.GAUGE_RADIUS_KM:g} km in the last {minutes} min"
    if name in C.PRIOR_RAIN and _known(value) and value >= 1:
        hours = int(name.removeprefix("rain_prior_").removesuffix("h_mm"))
        return f"{value:.0f} mm of rain already in the {hours - 1} hours before the last one"
    if name in NEARBY_RAIN and _known(value) and value >= 1:
        near, far = C.NEARBY_RAIN_KM
        where = "upwind, " if name == "rain_upwind_30min_mm" else ""
        return (
            f"{value:.0f} mm of rain {where}{near:g}-{far:g} km away in the last "
            f"{C.NEARBY_RAIN_MINUTES} min"
        )
    if name in C.DRAIN_RISE and _known(value) and value >= 1:
        # the drains' flow above its highest earlier in the storm: the last 10-30 minutes' rain is the storm's heaviest
        return "rain here is the heaviest it has been in this storm"
    if name in C.PUB_ALERTS and _known(value) and value >= 1:
        km, minutes = C.NEARBY_ALERT_WINDOWS[C.PUB_ALERTS.index(name)]
        alerts = "a PUB alert" if value < 2 else f"{value:.0f} PUB alerts"
        return f"{alerts} within {km} km in the last {minutes} min"
    if name == "forecast_rain_class" and value:
        return f"NEA forecasts {C.RAIN_CLASSES[int(value)]} here"
    if name == "tide_height_m" and value is not None and value >= HIGH_TIDE_M:
        return f"the tide is high ({value:.1f} m), so drains empty to the sea slowly"
    if (
        name == "minutes_to_high_tide"
        and value is not None
        and value <= C.HORIZON_MINUTES
    ):
        return f"high tide in {value:.0f} min"
    if name == AREA:
        return f"PUB has often alerted in {row[AREA]} in weather like this"
    return None


def _known(value) -> bool:
    """A value of a row that isn't null: None from Spark, NaN (of any float width) from pandas."""
    return value is not None and not pd.isna(value)
