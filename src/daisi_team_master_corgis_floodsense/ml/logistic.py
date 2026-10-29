"""Penalized logistic regression with offsets, sample weights and sign constraints: the linear learner a rung of the
ordinal model can use (ordinal.LOGISTIC) instead of LightGBM.

Why a linear model at all. The flood rung learns from about 180 rows a flash flood followed, from some 15 storms (gold,
23 Oct). A tree ensemble has little to grow on there, and a ridge-penalized logistic regression on a few transformed
inputs is the usual choice for so few events: stable, and close to calibrated. It is also the plain baseline the trees
must beat.

The inputs are the trees' (ordinal.design), transformed so that a straight line in the log-odds can follow them:

- rain (the windows near the area, the hours before them, the rain around the area and upwind) as log(1 + mm): the risk
  climbs steeply out of a dry spell and flattens in a downpour;
- the SGT hour as two harmonics (the sine and cosine of 2 pi h / 24 and of 4 pi h / 24), so midnight meets midnight;
- the rest as they are;
- then standardized by the training rows' weighted mean and spread. A missing value of an optional input (the weather,
  the upwind rain in a calm, the prior rain where a gauge missed an hour) becomes the mean, and an `<input>_missing`
  column says it was missing.

The fit minimizes the weighted log-loss plus penalty / 2 times the squared coefficients (the intercept is free), each
row's offset (its area prior) added to its log-odds, by L-BFGS-B. The weights are the population's (the quiet rows'
sampling weights), so the penalty counts in rows: against a few thousand alert rows, 1 is mild and 100 strong. The
coefficients of the inputs the chance may only rise with (gated.RISING) are bounded below by 0, which keeps the model
monotone in them as the trees are, their transforms being increasing. `out_of_fold` chooses the penalty from PENALTIES
by the folds' mean validation log-loss, as LightGBM's rounds are chosen.

Scoring needs only numpy, as for the trees: scipy is imported by the fit alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contract as C
from .credibility import area_offsets

HOUR = "hour_sgt"
# inputs read as log(1 + mm)
LOG_RAIN = (
    *(C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES),
    *C.PRIOR_RAIN,
    "rain_nearby_30min_mm",
    "rain_upwind_30min_mm",
)
PENALTIES = (0.1, 1.0, 10.0, 100.0)
MISSING = "_missing"
_CHUNK = 500_000  # rows scored at once, so scoring ten million rows holds one chunk's matrix


@dataclass
class Logistic:
    """A fitted model: its inputs, the columns it makes of them (`source[j]` is the input column j comes from), their
    centring and scale, and the coefficients."""

    inputs: tuple[str, ...]
    columns: tuple[str, ...]
    source: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    penalty: float

    def matrix(self, X: pd.DataFrame) -> np.ndarray:
        """The standardized columns of rows X (missing values at the mean, which is 0), as float32."""
        return _standardized(expand(X, self.inputs), self.columns, self.mean, self.scale)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        """Log-odds without the offsets."""
        out = np.empty(len(X))
        for lo in range(0, len(X), _CHUNK):
            part = X.iloc[lo : lo + _CHUNK]
            out[lo : lo + len(part)] = self.intercept + self.matrix(part) @ self.coef.astype(np.float32)
        return out

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per row, each input's share of the log-odds (its columns' coefficient times value, about the training mean),
        then the intercept last, as LightGBM's pred_contrib lays them out."""
        Z = self.matrix(X) * self.coef
        out = np.zeros((len(X), len(self.inputs) + 1))
        index = {c: i for i, c in enumerate(self.inputs)}
        for j, source in enumerate(self.source):
            out[:, index[source]] += Z[:, j]
        out[:, -1] = self.intercept
        return out

    def importance(self) -> pd.Series:
        """Per input, the sum of its columns' absolute coefficients (the columns are standardized)."""
        return (
            pd.Series(np.abs(self.coef), index=list(self.source))
            .groupby(level=0)
            .sum()
            .reindex(list(self.inputs), fill_value=0.0)
        )

    def to_dict(self) -> dict:
        return {
            "inputs": list(self.inputs),
            "columns": list(self.columns),
            "source": list(self.source),
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
            "penalty": float(self.penalty),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Logistic:
        return cls(
            inputs=tuple(d["inputs"]),
            columns=tuple(d["columns"]),
            source=tuple(d["source"]),
            mean=np.asarray(d["mean"]),
            scale=np.asarray(d["scale"]),
            coef=np.asarray(d["coef"]),
            intercept=d["intercept"],
            penalty=d["penalty"],
        )


def expand(X: pd.DataFrame, inputs: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The model's columns from the rows' inputs, before standardizing (module docstring)."""
    out = {}
    for c in inputs:
        v = X[c].to_numpy(dtype=float)
        if c == HOUR:
            for k in (1, 2):
                angle = 2 * np.pi * k * v / 24
                out[f"hour_sin{k}"], out[f"hour_cos{k}"] = np.sin(angle), np.cos(angle)
            continue
        out[c] = np.log1p(np.clip(v, 0, None)) if c in LOG_RAIN else v
        if c in C.OPTIONAL_FEATURES:
            out[c + MISSING] = np.isnan(v).astype(float)
    return out


def _standardized(raw: dict[str, np.ndarray], columns, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The columns stacked as float32, centred and scaled in place, missing values at the mean (0)."""
    Z = np.empty((len(next(iter(raw.values()))) if raw else 0, len(columns)), dtype=np.float32)
    for j, c in enumerate(columns):
        Z[:, j] = raw[c]
    Z -= mean.astype(np.float32)
    Z /= scale.astype(np.float32)
    return np.nan_to_num(Z, copy=False, nan=0.0)


def _sources(columns: list[str]) -> list[str]:
    return [
        HOUR if c.startswith("hour_") else c.removesuffix(MISSING) for c in columns
    ]


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
    *,
    penalty: float,
    rising=frozenset(),
    start: Logistic | None = None,
) -> Logistic:
    """Fit on the rows with weight > 0. `offset` is each row's log-odds offset (its area prior); `rising` names the
    inputs whose coefficients may not be negative; `start` warm-starts from another fit with the same columns."""
    from scipy.optimize import minimize
    from scipy.special import expit

    keep = weight > 0
    inputs = tuple(X.columns)
    raw = expand(X[keep], inputs)
    columns = list(raw)
    source = _sources(columns)
    w, yk, off = weight[keep], y[keep].astype(float), offset[keep]
    # weighted mean and spread of each column over its known values, one column at a time (no rows x columns copies)
    mean, scale = np.zeros(len(columns)), np.ones(len(columns))
    for j, c in enumerate(columns):
        v = raw[c]
        known = ~np.isnan(v)
        total = float(w[known].sum())
        if total > 0:
            mean[j] = float(np.dot(w[known], v[known])) / total
            spread = np.sqrt(float(np.dot(w[known], (v[known] - mean[j]) ** 2)) / total)
            scale[j] = spread if spread > 1e-9 else 1.0
    Z = _standardized(raw, columns, mean, scale)
    del raw

    w_sum = float(w.sum())

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        # the products in float32, as Z is (a float64 operand would copy all of Z each call), the sums in float64
        b, beta = theta[0], theta[1:]
        eta = off + b + (Z @ beta.astype(np.float32)).astype(float)
        loss = float(np.sum(w * (np.logaddexp(0.0, eta) - yk * eta)) + 0.5 * penalty * beta @ beta)
        r = w * (expit(eta) - yk)
        grad = np.concatenate([[r.sum()], (Z.T @ r.astype(np.float32)).astype(float) + penalty * beta])
        return loss / w_sum, grad / w_sum

    if start is not None and list(start.columns) == columns:
        theta0 = np.concatenate([[start.intercept], start.coef])
    else:
        rate = float(np.clip(np.average(yk, weights=w), 1e-9, 1 - 1e-9))
        b0 = np.log(rate / (1 - rate)) - float(np.average(off, weights=w))
        theta0 = np.concatenate([[b0], np.zeros(len(columns))])
    bounds = [(None, None)] + [
        (0.0, None) if (s in rising and not c.endswith(MISSING)) else (None, None)
        for c, s in zip(columns, source)
    ]
    theta0[1:] = [max(t, 0.0) if lo == 0.0 else t for t, (lo, _) in zip(theta0[1:], bounds[1:])]
    result = minimize(
        objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}
    )
    return Logistic(
        inputs=inputs,
        columns=tuple(columns),
        source=tuple(source),
        mean=mean,
        scale=scale,
        coef=result.x[1:].copy(),
        intercept=float(result.x[0]),
        penalty=float(penalty),
    )


def log_loss(y: np.ndarray, logit: np.ndarray, weight: np.ndarray) -> float:
    """Weighted mean log-loss of log-odds `logit`."""
    return float(np.average(np.logaddexp(0.0, logit) - y * logit, weights=weight))


def out_of_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    ts: pd.Series,
    area: pd.Series,
    fold_offsets: list[dict[str, float]],
    split,
    *,
    rising=frozenset(),
    penalties: tuple[float, ...] = PENALTIES,
) -> tuple[np.ndarray, float]:
    """Out-of-fold probabilities for every row, and the penalty the folds agree on (the lowest mean validation
    log-loss, over each block's sampled rows with their weights). Each fold fits on its training rows of the sample, its
    offsets its own area prior (computed without its block), and scores every row of its block. The penalties are tried
    from the strongest down, each fold starting from its previous fit."""
    keep = weight > 0
    fits: dict[float, list[Logistic]] = {}
    losses: dict[float, float] = {}
    previous: list[Logistic | None] = [None] * len(split.folds)
    for penalty in sorted(penalties, reverse=True):
        fold_fits, fold_losses = [], []
        for i, (fold, offsets) in enumerate(zip(split.folds, fold_offsets)):
            offset = area_offsets(area, offsets)
            train = fold.train(ts) & keep
            model = fit(
                X[train],
                y[train],
                weight[train],
                offset[train],
                penalty=penalty,
                rising=rising,
                start=previous[i],
            )
            valid = fold.valid(ts) & keep
            if valid.any():
                logit = model.raw(X[valid]) + offset[valid]
                fold_losses.append(log_loss(y[valid], logit, weight[valid]))
            fold_fits.append(model)
        previous = fold_fits
        fits[penalty], losses[penalty] = fold_fits, float(np.mean(fold_losses))
    best = min(losses, key=lambda p: (losses[p], -p))  # a tie goes to the stronger penalty
    scores = np.full(len(y), np.nan)
    for fold, model, offsets in zip(split.folds, fits[best], fold_offsets):
        block = fold.valid(ts)
        if block.any():
            logit = model.raw(X[block]) + area_offsets(area[block], offsets)
            scores[block] = 1.0 / (1.0 + np.exp(-logit))
    if np.isnan(scores).any():
        raise ValueError("some development rows fell in no fold's block")
    return scores, best
