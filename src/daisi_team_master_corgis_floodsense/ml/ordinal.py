"""The ordinal model: Low < Medium < High as two rungs, each a binary model fitted on every recorded area-tick, with no
gate in front (Frank and Hall's decomposition, 2001). It succeeds the gated LightGBM pair (gated.py) from 23 Oct.

Why. The pair's gate screened area-ticks out before its rater saw them: v5's rater learned from the 5.2% the gate
passed, and an alert tick the gate screened out could never be warned (at recall 0.9 the gate gave up a tenth of the
alert rows and 6.7% of the flood rows by design). And the pair learned one thing, the chance of any PUB alert within the hour; High was a higher
cut-off on that same chance, so nothing learned what makes a flash flood rather than a warning, which is where the
rainfall rule beat it (23 Oct: of 13 development floods, the rule warned 9 and the pair 6).

The ladder. A flash flood is an alert, so the labels are nested, no alert < a warning < a flash flood:

- the alert rung: P(a PUB alert of either kind within the hour), learned from alert_label: the rung to Medium;
- the flood rung: P(a flash flood within the hour), learned from flood_label: the rung to High. A warning (a drain at
  90% of capacity) is the near miss it learns to tell from a flood.

Fitting, on the development period of split.make_split (the blocked day-folds with an embargo either side, as the pair):

1. Rows. Every alert row, and the quiet rows sampled in two strata, each weighted back up by its share (`stormy`): the
   stormy ones (rain within 20 km in the last hour, or rain forecast: 31% of the rows, holding 97.7% of the alert rows
   and 98.3% of the flood rows; gold, 23 Oct) at `storm_quiet_share`, the dry ones at `dry_quiet_share`. That keeps
   what the gate was for, most of the learning going to the storms, without its cut: every row stays in reach.
2. Class weights, the cost of each miss: a row a flash flood followed weighs `flood_weight`, a near miss (a warning) and
   a quiet row 1. Both rungs train with them, and the alert rung's prior counts a flood day `flood_weight` times.
3. Each rung starts from its own area prior (credibility.py, as an offset): the alert rung's from the areas' alert
   days, the flood rung's from their flood days, which are so few that most areas get the island's rate, so an area
   with no water-level sensor, where PUB can't warn, isn't taken for one where nothing floods. Then LightGBM (the chance
   only rising with the rain at hand, around the area, the forecast and the tide, as in the pair) or a penalized
   logistic regression (logistic.py), out of fold.
4. Calibration on the pooled out-of-fold scores: centered isotonic for the alert rung, as the pair's rater; Platt
   scaling for the flood rung, two parameters, as some 15 storms can't support a step function. The flood chance is
   then capped at the alert chance (a flood is an alert).
5. The warning score, p_alert + (flood_value - 1) p_flood: what a warning at the tick is worth, a flood flagged counting
   flood_value events as when cut-offs are chosen (scoreboard.FLOOD_EVENT_VALUE). Medium is a cut-off on it, so a tick
   is warned for its flood risk as well as its alert risk. High is a cut-off on p_flood among the warned ticks: the
   lowest whose High warnings are followed by a flash flood often enough (scoreboard.TARGETS), else never. Both are
   chosen by scoreboard.score on the out-of-fold chances within the false-alarm budget, as the pair's are.
6. Both rungs are refitted on the whole development period, with the rounds (or the penalty) the folds chose and their
   priors from all of it. Every out-of-fold score comes from a prior computed without its own block.
7. Optionally, the rainfall rule as a safety override (`add_override`; train.py tries it on the chosen fit): a tick whose
   rain passes a threshold on the rule's window is at least Medium, whatever the model says. The threshold and the
   Medium cut-off are chosen together out of fold, within the same false-alarm budget, and the override is kept only if
   it flags more.

Scoring a row (`predict`): a row missing a required feature isn't rated (no chance, tier Low); an optional one missing
is read as missing. `chances` gives p_alert, p_flood and the warning score, `tiers` the tier, and `reasons` the app's
plain English: the two inputs that raised the warned rung's score most (the flood rung for High, the alert rung for
Medium), or the rule's words where only the override warned. ml.predictions keeps p_alert, the chance of an alert, as
for the pair. `save` and `load` keep the model as ordinal.json and a LightGBM file per tree rung.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import contract as C
from . import gated, logistic
from .credibility import ALERT_CLAIMS, FLOOD_CLAIMS, area_days, area_offsets, area_prior
from .scoreboard import FLOOD_EVENT_VALUE, TARGETS, score
from .split import Split, _sgt

AREA, HOUR = gated.AREA, gated.HOUR
ALERT, FLOOD = "alert", "flood"
RUNGS = (ALERT, FLOOD)
LIGHTGBM, LOGISTIC, GATED = "lightgbm", "logistic", "gated"
LEARNERS = (LIGHTGBM, LOGISTIC)
# the alert rung can also be the gated pair (gated.py): its gate as hard-negative mining for its rater, without its veto
ALERT_LEARNERS = (*LEARNERS, GATED)
BASE, EXTENDED = "base", "extended"
FEATURE_SETS = {BASE: C.BASE_FEATURES, EXTENDED: C.EXTENDED_FEATURES}
SAVED = "ordinal.json"
ISOTONIC, PLATT = "isotonic", "platt"
# the override's candidate thresholds, as multiples of the rule's Medium threshold (snapped to its 2.5 mm grid)
OVERRIDE_MULTIPLES = (1.0, 1.5, 2.0, 3.0)

# The alert rung has the rater's settings (some 4,000 alert rows). The flood rung has some 180 rows to learn from, so
# fewer leaves, more rows per leaf and more shrinkage.
ALERT_PARAMS = gated.RATER_PARAMS
FLOOD_PARAMS = {
    **gated.RATER_PARAMS,
    "num_leaves": 7,
    "min_data_in_leaf": 200,
    "lambda_l2": 10.0,
    "feature_fraction": 0.8,
}


@dataclass(frozen=True)
class Settings:
    """alert_learner, flood_learner: LIGHTGBM or LOGISTIC per rung, or GATED for the alert rung (the gated pair at
    `gate_recall`). features: BASE (contract.BASE_FEATURES, what the gated pair v5 reads) or EXTENDED (the dew point
    for the humidity, and the prior rain), less `drop`. flood_weight: what a row a flash flood followed weighs
    against a near miss or a quiet row. The quiet shares: how many quiet rows each stratum keeps, weighted back up.
    max_warned_share: the most rows a Medium cut-off candidate may warn, as more is far past any false-alarm budget.
    The rest as in gated.Settings."""

    alert_learner: str = LIGHTGBM
    flood_learner: str = LIGHTGBM
    gate_recall: float = gated.DEFAULTS.gate_recall
    features: str = BASE
    flood_weight: float = 1.0
    flood_value: float = FLOOD_EVENT_VALUE
    storm_quiet_share: float = 0.2
    dry_quiet_share: float = 0.02
    max_rounds: int = 2000
    patience: int = 100
    max_false_alarms: float = 1.0
    min_high_precision: float = TARGETS["high_precision"]
    cut_off_candidates: int = 60
    max_warned_share: float = 0.1
    seed: int = 0
    drop: tuple[str, ...] = ()
    # model features read beyond the feature set (train.py's add=, e.g. contract.PUB_ALERTS)
    add: tuple[str, ...] = ()

    def __post_init__(self):
        if self.alert_learner not in ALERT_LEARNERS or self.flood_learner not in LEARNERS:
            raise ValueError(f"the alert rung's learner is one of {ALERT_LEARNERS}, the flood rung's of {LEARNERS}")
        if self.features not in FEATURE_SETS:
            raise ValueError(f"features is one of {tuple(FEATURE_SETS)}, not {self.features!r}")
        if self.flood_weight < 1 or self.flood_value < 1:
            raise ValueError("flood_weight and flood_value are at least 1: a flood weighs no less than a warning")
        unknown = (set(self.drop) | set(self.add)) - set(C.MODEL_FEATURES)
        if unknown:
            raise ValueError(f"not model features: {sorted(unknown)}")

    @property
    def inputs(self) -> tuple[str, ...]:
        """The hour and the set's features with `add`'s, less `drop`'s, in gold's order (as the gated pair keeps it)."""
        read = {*FEATURE_SETS[self.features], *self.add}
        return (HOUR, *(c for c in C.MODEL_FEATURES if c in read and c not in self.drop))

    def learner(self, rung: str) -> str:
        return self.alert_learner if rung == ALERT else self.flood_learner

    def pair_settings(self) -> gated.Settings:
        """The gated pair's settings when it is the alert rung: these inputs, budget, weights and fitting limits."""
        return gated.Settings(
            gate_recall=self.gate_recall,
            flood_weight=self.flood_weight,
            flood_value=self.flood_value,
            max_rounds=self.max_rounds,
            patience=self.patience,
            max_false_alarms=self.max_false_alarms,
            min_high_precision=self.min_high_precision,
            cut_off_candidates=self.cut_off_candidates,
            seed=self.seed,
            drop=tuple(c for c in C.MODEL_FEATURES if c not in self.inputs),
        )


DEFAULTS = Settings()


def design(frame: pd.DataFrame, inputs: tuple[str, ...]) -> pd.DataFrame:
    """The rungs' inputs from gold rows: the SGT hour of day and the model features of `inputs`, in its order."""
    ts = _sgt(frame["ts"])
    columns = {
        HOUR: (ts.dt.hour + ts.dt.minute / 60).astype("float32").to_numpy(),
        **{c: pd.to_numeric(frame[c]).astype("float32").to_numpy() for c in inputs if c != HOUR},
    }
    return pd.DataFrame(columns, index=frame.index)[list(inputs)]


def stormy(frame: pd.DataFrame) -> np.ndarray:
    """The rows with rain about: rain within 20 km in the last hour, or a rain forecast (the stratum the quiet rows are
    sampled most densely in). The rule-like gate a recall near 1 would need: it passes 31% of the rows (23 Oct)."""
    return (
        (frame["rain_60min_mm"] > 0)
        | (frame["rain_nearby_30min_mm"] > 0)
        | (frame["forecast_rain_class"] >= 1)
    ).to_numpy()


def sample_weight(frame: pd.DataFrame, alert: np.ndarray, settings: Settings) -> np.ndarray:
    """Each row's sampling weight: 1 for an alert row, 1 / share for a quiet row kept in its stratum, 0 for one left out
    (so totals stay unbiased)."""
    storm = stormy(frame)
    share = np.where(storm, settings.storm_quiet_share, settings.dry_quiet_share)
    kept = np.random.default_rng(settings.seed).random(len(frame)) < share
    return np.where(alert == 1, 1.0, np.where(kept, 1.0 / share, 0.0))


@dataclass
class Rung:
    """One rung: its learner's model, the log-odds it starts from besides the area offsets (`base`, LightGBM's; a
    logistic model's intercept holds its own), the area prior (log relative risk per area) and the calibration of its
    raw chance (isotonic: x, y on the probability; platt: a, b on the log-odds). A GATED rung is a whole gated pair,
    which brings its own prior, gate and calibration."""

    learner: str
    model: object  # lgb.Booster, logistic.Logistic or gated.GatedPair
    base: float
    area_offsets: dict[str, float]
    calibration: dict

    def logit(self, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        if self.learner == LIGHTGBM:
            return self.model.predict(X, raw_score=True) + self.base + offset
        return self.model.raw(X) + offset

    def p(self, X: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        """The calibrated chance of rows (gold rows, all their required features known; X their design)."""
        if self.learner == GATED:
            return self.model.p_alert(rows)[1]
        return calibrated(self.logit(X, area_offsets(rows[AREA], self.area_offsets)), self.calibration)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per row, each input's share of the raw log-odds, then the bias (the area offset is not among them). For a
        GATED rung, its rater's (the pair reads the same inputs in the same order)."""
        if self.learner == LIGHTGBM:
            return self.model.predict(X, pred_contrib=True)
        if self.learner == GATED:
            return self.model.rater.predict(X, pred_contrib=True)
        return self.model.contributions(X)

    def importance(self, inputs: tuple[str, ...]) -> np.ndarray:
        """Each input's share of the rung's importance: LightGBM's gain (a GATED rung's rater's), or the logistic
        model's absolute coefficients on standardized columns."""
        if self.learner == LIGHTGBM:
            v = np.asarray(self.model.feature_importance("gain"), dtype=float)
        elif self.learner == GATED:
            v = np.asarray(self.model.rater.feature_importance("gain"), dtype=float)
        else:
            v = self.model.importance().reindex(list(inputs)).to_numpy(dtype=float)
        return v / max(v.sum(), np.finfo(float).tiny)


def calibrated(logit: np.ndarray, calibration: dict) -> np.ndarray:
    if calibration["kind"] == ISOTONIC:
        return np.interp(_sigmoid(logit), calibration["x"], calibration["y"])
    return _sigmoid(calibration["a"] * logit + calibration["b"])


@dataclass
class OrdinalModel:
    """The fitted ladder: both rungs, the Medium cut-off on the warning score, the High cut-off on p_flood, and the
    override (the rule's window in minutes and threshold in mm), if any. `predict` writes ml.predictions rows."""

    areas: tuple[str, ...]
    inputs: tuple[str, ...]
    alert: Rung
    flood: Rung
    medium: float
    high: float = math.inf
    flood_value: float = FLOOD_EVENT_VALUE
    override: tuple[int, float] | None = None
    flood_weight: float = 1.0

    @property
    def features(self) -> list[str]:
        return [c for c in self.inputs if c in C.MODEL_FEATURES]

    @property
    def name(self) -> str:
        over = f" R>={self.override[1]:g}mm/{self.override[0]}min" if self.override else ""
        return f"ordinal {self.alert.learner}/{self.flood.learner} M>={self.medium:.3g} H>={self.high:.3g}{over}"

    @property
    def area_priors(self) -> dict[str, dict[str, float]]:
        return {ALERT: self.alert.area_offsets, FLOOD: self.flood.area_offsets}

    def rung(self, name: str) -> Rung:
        return self.alert if name == ALERT else self.flood

    def chances(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per row: p_alert, p_flood (at most p_alert) and the warning score, NaN where a required feature is missing;
        and `known`, where none is."""
        X = design(frame, self.inputs)
        required = [c for c in C.REQUIRED_FEATURES if c in self.features]
        known = X[required].notna().all(axis=1).to_numpy()
        p_alert, p_flood = np.full(len(X), np.nan), np.full(len(X), np.nan)
        if known.any():
            rows, Xk = frame[known], X[known]
            p_alert[known] = self.alert.p(Xk, rows)
            p_flood[known] = np.minimum(self.flood.p(Xk, rows), p_alert[known])
        return {
            "p_alert": p_alert,
            "p_flood": p_flood,
            "score": warning_score(p_alert, p_flood, self.flood_value),
            "known": known,
        }

    def forced(self, frame: pd.DataFrame) -> np.ndarray:
        """Where the override warns: the rule's window at or above its threshold."""
        return gated.override_mask(frame, self.override)

    def tiers(self, chances: dict[str, np.ndarray], frame: pd.DataFrame) -> np.ndarray:
        warned = chances["known"] & ((chances["score"] >= self.medium) | self.forced(frame))
        return _tiers(warned, chances["p_flood"], self.high)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rows for ml.predictions, with the reason the app shows for every warning."""
        chances = self.chances(frame)
        tier = self.tiers(chances, frame)
        return pd.DataFrame(
            {
                "model": self.name,
                "area": frame[AREA],
                "ts": frame["ts"],
                "tier": tier,
                "p_alert": chances["p_alert"],
                "reason": self.reasons(frame, tier, chances),
            }
        )

    def reasons(self, frame: pd.DataFrame, tier: np.ndarray, chances: dict[str, np.ndarray]) -> pd.Series:
        """Plain English for each warned row: the two inputs that raised the warned rung's score most (the flood rung
        for High, the alert rung for Medium; the area prior beside them), or the rule's words where only the override
        warned."""
        reason = pd.Series(None, index=frame.index, dtype=object)
        warned = np.asarray(tier) != "Low"
        if not warned.any():
            return reason
        rows = frame[warned]
        X = design(rows, self.inputs)
        names = [*self.inputs, AREA]
        contributions = {}
        for name in RUNGS:
            rung = self.rung(name)
            c = rung.contributions(X)
            # the prior shifts the score as an input's contribution would: put it beside them, before the bias
            contributions[name] = np.column_stack(
                [c[:, :-1], area_offsets(rows[AREA], rung.area_offsets), c[:, -1]]
            )
        model_warned = chances["score"][warned] >= self.medium
        high = np.asarray(tier)[warned] == "High"
        texts = []
        for i, row in enumerate(rows.to_dict("records")):
            if not model_warned[i]:
                texts.append(gated.override_reason(row, self.override))
                continue
            contribution = contributions[FLOOD if high[i] else ALERT][i]
            ranked = np.argsort(-contribution[:-1])  # the last column is the bias
            texts.append(gated.reason_text([names[j] for j in ranked if contribution[j] > 0], row))
        reason[warned] = texts
        return reason

    def logged_cut_offs(self) -> dict[str, float]:
        """The cut-offs as train.py logs them (model_*), which verify.py checks the registered model against."""
        out = {"model_medium_cut": self.medium}
        if self.alert.learner == GATED:
            out["model_gate_threshold"] = self.alert.model.gate_threshold
        if math.isfinite(self.high):
            out["model_high_cut"] = self.high
        if self.override is not None:
            out["model_override_mm"] = self.override[1]
        return out

    def save(self, directory) -> None:
        """ordinal.json, and a LightGBM model file per tree rung, so loading needs only lightgbm and numpy."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        rungs = {}
        for name in RUNGS:
            rung = self.rung(name)
            entry = {
                "learner": rung.learner,
                "base": rung.base,
                "area_offsets": {a: float(v) for a, v in rung.area_offsets.items()},
                "calibration": _jsonable(rung.calibration),
            }
            if rung.learner == LIGHTGBM:
                rung.model.save_model(str(out / f"{name}.txt"))
            elif rung.learner == GATED:
                rung.model.save(out / f"{name}_pair")
            else:
                entry["logistic"] = rung.model.to_dict()
            rungs[name] = entry
        spec = {
            "kind": "ordinal",
            "inputs": list(self.inputs),
            "areas": list(self.areas),
            "medium": self.medium,
            "high": None if math.isinf(self.high) else self.high,
            "flood_value": self.flood_value,
            "override": None if self.override is None else {"window_min": self.override[0], "mm": self.override[1]},
            "flood_weight": self.flood_weight,
            "rungs": rungs,
        }
        (out / SAVED).write_text(json.dumps(spec, indent=1))

    @classmethod
    def load(cls, directory) -> OrdinalModel:
        src = Path(directory)
        s = json.loads((src / SAVED).read_text())
        inputs = tuple(s["inputs"])
        unknown = set(inputs) - {HOUR, *C.MODEL_FEATURES}
        if unknown:
            raise ValueError(f"saved for inputs this code doesn't know: {sorted(unknown)}")
        rungs = {}
        for name in RUNGS:
            e = s["rungs"][name]
            if e["learner"] == LIGHTGBM:
                model = lgb.Booster(model_file=str(src / f"{name}.txt"))
            elif e["learner"] == GATED:
                model = gated.GatedPair.load(src / f"{name}_pair")
            else:
                model = logistic.Logistic.from_dict(e["logistic"])
            rungs[name] = Rung(e["learner"], model, e["base"], e["area_offsets"], e["calibration"])
        over = s["override"]
        return cls(
            areas=tuple(s["areas"]),
            inputs=inputs,
            alert=rungs[ALERT],
            flood=rungs[FLOOD],
            medium=s["medium"],
            high=math.inf if s["high"] is None else s["high"],
            flood_value=s["flood_value"],
            override=None if over is None else (int(over["window_min"]), float(over["mm"])),
            flood_weight=s.get("flood_weight", 1.0),
        )


@dataclass
class Fit:
    """What fitting learned, out of fold, for logging: one entry per development row in `rows` order."""

    rows: pd.DataFrame  # area, ts, alert_label, flood_label
    p_alert: np.ndarray
    p_flood: np.ndarray
    score: np.ndarray
    tier: np.ndarray
    curve: pd.DataFrame  # every Medium candidate's out-of-fold scores
    stats: dict[str, float] = field(default_factory=dict)
    importance: pd.DataFrame | None = None
    area_prior: pd.DataFrame | None = None  # both rungs' priors (credibility.AreaPrior.table), with a rung column

    def predictions(self, *, warned_only: bool = True) -> pd.DataFrame:
        """Out-of-fold (area, ts, tier, p_alert) rows, as scoreboard.score takes them."""
        out = self.rows[[AREA, "ts"]].assign(tier=self.tier, p_alert=self.p_alert)
        return out[self.tier != "Low"] if warned_only else out

    def reliability(self, bins: int = 8) -> pd.DataFrame:
        """Calibration of each rung out of fold: mean chance against the rate, in bins of the chance whose edges fall
        on a geometric range of the top shares of the rows (most rows are dry and near 0, the useful range is the
        tail)."""
        out = []
        for name, p, y in (
            (ALERT, self.p_alert, self.rows["alert_label"].to_numpy()),
            (FLOOD, self.p_flood, self.rows["flood_label"].to_numpy()),
        ):
            edges = np.unique(np.quantile(p, [0.0, *(1 - np.geomspace(0.5, 1e-5, bins)), 1.0]))
            group = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            table = pd.DataFrame({"g": group, "p": p, "y": y}).groupby("g")
            agg = table.agg(rows=("p", "size"), mean_p=("p", "mean"), rate=("y", "mean")).reset_index()
            agg.insert(0, "rung", name)
            agg.insert(1, "p_range", [f"{edges[g]:.3g}-{edges[g + 1]:.3g}" for g in agg.pop("g")])
            out.append(agg)
        return pd.concat(out, ignore_index=True)


def warning_score(p_alert: np.ndarray, p_flood: np.ndarray, flood_value: float) -> np.ndarray:
    """What a warning at the tick is worth: the chance of an alert, plus flood_value - 1 times the chance of a flood (a
    flood flagged counts flood_value events, as gated.weighted_flagged counts it)."""
    return p_alert + (flood_value - 1.0) * p_flood


def fit(
    rows: pd.DataFrame,
    alerts: pd.DataFrame,
    split: Split,
    *,
    missing=None,
    blind=None,
    settings: Settings = DEFAULTS,
) -> tuple[OrdinalModel, Fit]:
    """Fit the ladder on the development period of `split` (module docstring). The arguments are gated.fit's."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score

    rows = gated.development_rows(rows, split)
    areas = tuple(sorted(rows[AREA].unique()))
    labels = {
        ALERT: rows["alert_label"].to_numpy(dtype=np.int8),
        FLOOD: rows["flood_label"].to_numpy(dtype=np.int8),
    }
    if not labels[ALERT].any() or not labels[FLOOD].any():
        raise ValueError("the development rows need alerts and flash floods for both rungs to learn from")
    ts, area = rows["ts"], rows[AREA]
    flood_row = labels[FLOOD] == 1
    weight = sample_weight(rows, labels[ALERT], settings) * np.where(flood_row, settings.flood_weight, 1.0)
    days = area_days(rows)
    day = pd.Series(days["day"])
    rising = gated.RISING & set(settings.inputs)

    oof, rungs, priors, chosen = {}, {}, [], {}
    X = None  # the rungs' design, built once a rung needs it (the gated pair builds its own)
    for name in RUNGS:
        if settings.learner(name) == GATED:
            # the gated pair as the alert rung, fitted out of fold on the same rows, folds and budget; its chance is
            # already calibrated, with the rows its gate screens out at its quiet rate
            pair, pair_fit = gated.fit(rows, alerts, split, missing=missing, blind=blind, settings=settings.pair_settings())
            oof[name] = pair_fit.p_alert
            rungs[name] = Rung(GATED, pair, 0.0, pair.area_offsets or {}, {"kind": GATED})
            k = pair_fit.stats.get("area_prior_k_days", math.nan)
            if pair_fit.area_prior is not None:
                priors.append(pair_fit.area_prior.assign(rung=name, k_days=k if math.isfinite(k) else math.nan))
            chosen |= {
                key: pair_fit.stats[key]
                for key in (
                    "gate_threshold", "gate_pass_share", "gate_alert_row_recall", "gate_flood_row_recall",
                    "gate_rounds", "rater_rounds", "quiet_p",
                )
            }  # fmt: skip
            del pair_fit
            continue
        if X is None:
            X = design(rows, settings.inputs)
        claims = ALERT_CLAIMS if name == ALERT else FLOOD_CLAIMS
        fold_offsets = [
            area_prior(days[fold.train(day)], flood_weight=settings.flood_weight, claims=claims).offsets
            for fold in split.folds
        ]
        prior = area_prior(days, flood_weight=settings.flood_weight, claims=claims)
        # k is infinite where the areas differ only by chance (every area at the island's rate): NaN in the table
        priors.append(prior.table.assign(rung=name, k_days=prior.k if math.isfinite(prior.k) else math.nan))
        y = labels[name]
        if settings.learner(name) == LIGHTGBM:
            params = ALERT_PARAMS if name == ALERT else FLOOD_PARAMS
            raw, rounds = gated._out_of_fold(X, y, weight, ts, area, fold_offsets, split, params, settings)
            model, base = gated._train(X, y, weight, area, prior.offsets, params, rounds, settings)
            chosen[f"{name}_rounds"] = rounds
        else:
            raw, penalty = logistic.out_of_fold(
                X, y, weight, ts, area, fold_offsets, split, rising=rising
            )
            model = logistic.fit(
                X, y, weight, area_offsets(area, prior.offsets), penalty=penalty, rising=rising
            )
            base = 0.0
            chosen[f"{name}_penalty"] = penalty
        logit = _logit(raw)
        if name == ALERT:
            isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
            x_, y_ = gated._centered(
                np.asarray(isotonic.X_thresholds_, dtype=float), np.asarray(isotonic.y_thresholds_, dtype=float)
            )
            calibration = {"kind": ISOTONIC, "x": x_, "y": y_}
        else:
            calibration = {"kind": PLATT, **_platt(logit, y)}
        oof[name] = calibrated(logit, calibration)
        rungs[name] = Rung(settings.learner(name), model, base, prior.offsets, calibration)
        del raw, logit
    del X  # 1 GB at ten million rows: free it before the cut-off search

    p_alert = oof[ALERT]
    p_flood = np.minimum(oof[FLOOD], p_alert)
    warn = warning_score(p_alert, p_flood, settings.flood_value)
    medium, high, curve = cut_offs(
        rows, warn, p_flood, alerts, split, missing=missing, blind=blind, settings=settings
    )
    model = OrdinalModel(
        areas=areas,
        inputs=settings.inputs,
        alert=rungs[ALERT],
        flood=rungs[FLOOD],
        medium=medium,
        high=high,
        flood_value=settings.flood_value,
        flood_weight=settings.flood_weight,
    )
    warned = warn >= medium
    tier = _tiers(warned, p_flood, high)
    storm = stormy(rows)
    stats = {
        "rows": len(rows),
        "alert_rows": int(labels[ALERT].sum()),
        "flood_rows": int(flood_row.sum()),
        "sampled_rows": int((weight > 0).sum()),
        "stormy_share": float(storm.mean()),
        # what the rule-like gate a recall near 1 needs would have passed: the stratum the quiet rows are sampled in
        "stormy_alert_row_recall": float(storm[labels[ALERT] == 1].mean()),
        "stormy_flood_row_recall": float(storm[flood_row].mean()),
        "flood_weight": settings.flood_weight,
        "flood_value": settings.flood_value,
        **{k: float(v) for k, v in chosen.items()},
        "medium_cut": medium,
        "high_cut": high,
        "warned_row_share": float(warned.mean()),
        "alert_row_recall": float(warned[labels[ALERT] == 1].mean()),
        "flood_row_recall": float(warned[flood_row].mean()),
        # threshold-free: average precision against each rung's base rate, which is what a random score gets
        "alert_pr_auc": float(average_precision_score(labels[ALERT], p_alert)),
        "flood_pr_auc": float(average_precision_score(labels[FLOOD], p_flood)),
        "alert_base_rate": float(labels[ALERT].mean()),
        "flood_base_rate": float(labels[FLOOD].mean()),
        "alert_brier": float(np.mean((p_alert - labels[ALERT]) ** 2)),
        "flood_brier": float(np.mean((p_flood - labels[FLOOD]) ** 2)),
        "brier": float(np.mean((p_alert - labels[ALERT]) ** 2)),
        "alert_prior_k_days": float(priors[0]["k_days"].iloc[0]) if len(priors) == 2 else math.nan,
        "flood_prior_k_days": float(priors[-1]["k_days"].iloc[0]),
        "flood_prior_offset_min": float(priors[-1]["offset"].min()),
        "flood_prior_offset_max": float(priors[-1]["offset"].max()),
    }
    importance = pd.DataFrame(
        {
            "input": settings.inputs,
            "alert_gain": model.alert.importance(settings.inputs),
            "flood_gain": model.flood.importance(settings.inputs),
        }
    ).sort_values("alert_gain", ascending=False)
    report = Fit(
        rows=rows[[AREA, "ts", "alert_label", "flood_label"]],
        p_alert=p_alert,
        p_flood=p_flood,
        score=warn,
        tier=tier,
        curve=curve,
        stats=stats,
        importance=importance,
        area_prior=pd.concat(priors, ignore_index=True),
    )
    return model, report


def cut_offs(
    rows: pd.DataFrame,
    warn: np.ndarray,
    p_flood: np.ndarray,
    alerts: pd.DataFrame,
    split: Split,
    *,
    missing=None,
    blind=None,
    settings: Settings = DEFAULTS,
    forced: np.ndarray | None = None,
    at_least: float = -math.inf,
) -> tuple[float, float, pd.DataFrame]:
    """Medium and High cut-offs on out-of-fold chances, as gated.cut_offs sets the pair's: Medium on the warning score
    `warn`, the candidate that flags the most PUB events before PUB within the false-alarm budget (a flood counting
    flood_value times), ties to the longer lead, then to fewer warned hours; High on `p_flood` among the warned rows,
    the lowest whose High warnings are followed by a flash flood often enough, else never. `forced` marks rows warned
    whatever the score (the override); `at_least` leaves out lower Medium candidates. Returns both and the Medium curve.
    """
    forced = np.zeros(len(rows), dtype=bool) if forced is None else forced
    frame = rows[[AREA, "ts"]]
    shares = np.geomspace(1e-5, settings.max_warned_share, settings.cut_off_candidates)
    candidates = [float(c) for c in np.unique(np.quantile(warn, 1 - shares)) if c >= at_least and c > 0]

    def metrics(medium: float, high: float = math.inf) -> dict[str, float]:
        warned = (warn >= medium) | forced
        tier = _tiers(warned, p_flood, high)
        board = score(
            frame[warned].assign(tier=tier[warned]),
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
                "weighted_flagged": gated.weighted_flagged(m, settings.flood_value),
                "row_share": float(((warn >= c) | forced).mean()),
            }
        )
    curve = pd.DataFrame(curve)
    within = curve[curve["false_alarms"] <= settings.max_false_alarms] if len(curve) else curve
    if within.empty:
        raise ValueError(f"every cut-off raises more than {settings.max_false_alarms} false alarms per area-month")
    best = within.sort_values(
        ["weighted_flagged", "lead_min", "warned_hours"], ascending=[False, False, True]
    ).iloc[0]
    medium = float(best["medium_p"])

    high = math.inf
    warned = (warn >= medium) | forced
    if warned.any():
        tail = np.geomspace(1e-3, 1, settings.cut_off_candidates // 2)
        for h in np.unique(np.quantile(p_flood[warned], 1 - tail)):
            if h <= 0:
                continue
            m = metrics(medium, float(h))
            if m["high_warnings"] and m["high_precision"] >= settings.min_high_precision:
                high = float(h)
                break
    return medium, high, curve


def add_override(
    model,
    report,
    dev: pd.DataFrame,
    alerts: pd.DataFrame,
    split: Split,
    rule,
    *,
    missing=None,
    blind=None,
    settings=DEFAULTS,
):
    """The rainfall rule as a safety override, if it earns its place (module docstring, 7), on an OrdinalModel or a
    gated.GatedPair and its out-of-fold Fit. `dev` is the development rows in report.rows order
    (gated.development_rows), `rule` the fitted baseline.RainfallRule. The model as fitted is the first option; each
    threshold on the rule's window (OVERRIDE_MULTIPLES of its Medium threshold) is tried with the Medium cut-off chosen
    again on top of it, no lower than without it (the override's warnings use the budget too); the best within the
    budget is kept, no override if none flags more. For the pair, Medium and High are cut-offs on its one chance, as
    gated.cut_offs sets them. Returns the model and its fit with the override chosen, and the table of thresholds."""
    pair = isinstance(model, gated.GatedPair)
    if pair:
        warn, high_score, medium0, high0 = report.p_alert, report.p_alert, model.medium_p, model.high_p
    else:
        warn, high_score, medium0, high0 = report.score, report.p_flood, model.medium, model.high
    limits = settings if isinstance(settings, Settings) else Settings(
        max_false_alarms=settings.max_false_alarms,
        flood_value=settings.flood_value,
        min_high_precision=settings.min_high_precision,
        cut_off_candidates=settings.cut_off_candidates,
    )
    judge = {"start": split.start, "end": split.dev_end, "missing": missing, "blind": blind}
    rain = pd.to_numeric(dev[rule.column]).to_numpy(dtype=float)
    thresholds = sorted({2.5 * round(rule.medium_mm * k / 2.5) for k in OVERRIDE_MULTIPLES})

    def summary(mm, medium, high, forced_share, m) -> dict:
        return {
            "override_mm": mm,
            "within_budget": True,
            "medium_cut": medium,
            "high_cut": high,
            "forced_row_share": forced_share,
            "flagged": m["alert_flagged_before_pub"],
            "flood_flagged": m["flood_flagged_before_pub"],
            "weighted_flagged": gated.weighted_flagged(m, limits.flood_value),
            "lead_min": m["alert_mean_lead_min"],
            "false_alarms": m["false_alarms_per_area_month"],
        }

    # the model as fitted, without an override: its own cut-offs and tiers
    options = [(summary(None, medium0, high0, 0.0, score(report.predictions(), alerts, **judge).metrics(ci=False)),
                report.tier)]  # fmt: skip
    table = [options[0][0]]
    for mm in thresholds:
        forced = rain >= mm
        try:
            medium, high, _ = cut_offs(
                report.rows, warn, high_score, alerts, split, missing=missing, blind=blind,
                settings=limits, forced=forced, at_least=medium0,
            )  # fmt: skip
        except ValueError:  # every cut-off over the budget with this override
            table.append({"override_mm": mm, "within_budget": False})
            continue
        warned = (warn >= medium) | forced
        tier = _tiers(warned, high_score, high)
        m = score(report.rows.loc[warned, [AREA, "ts"]].assign(tier=tier[warned]), alerts, **judge).metrics(ci=False)
        options.append((summary(mm, medium, high, float(forced.mean()), m), tier))
        table.append(options[-1][0])
    table = pd.DataFrame(table)

    def rank(row: dict) -> tuple:
        return (row["weighted_flagged"], row["lead_min"], -row["false_alarms"], row["override_mm"] is None)

    best, tier = max(options, key=lambda o: tuple(np.nan_to_num(rank(o[0]), nan=-math.inf)))
    mm = best["override_mm"]
    override = None if mm is None else (rule.window_min, float(mm))
    if pair:
        model = replace(model, medium_p=best["medium_cut"], high_p=best["high_cut"], override=override)
        cuts = {"medium_p": best["medium_cut"], "high_p": best["high_cut"]}
    else:
        model = replace(model, medium=best["medium_cut"], high=best["high_cut"], override=override)
        cuts = {"medium_cut": best["medium_cut"], "high_cut": best["high_cut"]}
    stats = {
        **report.stats,
        **cuts,
        "override_mm": math.nan if mm is None else float(mm),
        "override_row_share": best["forced_row_share"],
    }
    return model, replace(report, tier=tier, stats=stats), table


def _tiers(warned: np.ndarray, p_flood: np.ndarray, high: float) -> np.ndarray:
    return np.select([warned & (p_flood >= high), warned], ["High", "Medium"], "Low")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _platt(logit: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """P(y) = sigmoid(a logit + b) by maximum likelihood, a kept positive so the calibration never reverses the
    ranking."""
    from scipy.optimize import minimize
    from scipy.special import expit

    y = y.astype(float)

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] * logit + theta[1]
        r = expit(eta) - y
        n = len(y)
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta)) / n, np.array([r @ logit, r.sum()]) / n

    result = minimize(loss, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[(1e-3, None), (None, None)])
    return {"a": float(result.x[0]), "b": float(result.x[1])}


def _jsonable(calibration: dict) -> dict:
    return {k: ([float(x) for x in v] if isinstance(v, np.ndarray) else v) for k, v in calibration.items()}
