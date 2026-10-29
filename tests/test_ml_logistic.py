"""The penalized logistic regression a rung of the ordinal model can use: it recovers what it should, keeps the signs it
is told to, and scores from its saved form alone."""

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import logistic


def simulated(n: int = 60_000, seed: int = 0):
    """Rare events whose log-odds rise with log(1 + rain) and fall with a prior-rain input, an offset per row, and an
    optional input missing for a third of the rows."""
    rng = np.random.default_rng(seed)
    rain = rng.gamma(0.3, 8.0, n)
    prior = rng.gamma(0.5, 10.0, n)
    temp = np.where(rng.random(n) < 1 / 3, np.nan, rng.normal(28, 2, n))
    offset = rng.choice([-1.0, 0.0, 1.0], n)
    eta = -6.0 + 1.5 * np.log1p(rain) - 0.4 * np.log1p(prior) + offset
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(np.int8)
    X = pd.DataFrame({"rain_60min_mm": rain, "rain_prior_3h_mm": prior, "air_temp_c": temp})
    return X, y, offset


def test_it_recovers_the_slopes_and_reads_the_offset():
    X, y, offset = simulated()
    model = logistic.fit(X, y, np.ones(len(y)), offset, penalty=0.1)
    coef = dict(zip(model.columns, model.coef / model.scale))  # back to the transformed units
    assert coef["rain_60min_mm"] == pytest.approx(1.5, abs=0.25)
    assert coef["rain_prior_3h_mm"] == pytest.approx(-0.4, abs=0.2)
    assert abs(coef["air_temp_c"]) < 0.15  # no effect, and its missing rows got an indicator instead of a guess
    assert "air_temp_c_missing" in model.columns
    # its chance tracks the events
    logit = model.raw(X) + offset
    assert np.mean(1 / (1 + np.exp(-logit))) == pytest.approx(y.mean(), rel=0.1)


def test_a_rising_input_never_gets_a_negative_coefficient():
    X, y, offset = simulated()
    # told the prior rain may only raise the chance, the fit holds its coefficient at 0 rather than below
    model = logistic.fit(X, y, np.ones(len(y)), offset, penalty=0.1, rising={"rain_60min_mm", "rain_prior_3h_mm"})
    coef = dict(zip(model.columns, model.coef))
    assert coef["rain_prior_3h_mm"] == pytest.approx(0.0, abs=1e-9) and coef["rain_60min_mm"] > 0


def test_sample_weights_stand_for_the_rows_left_out():
    # keeping a tenth of the non-events at weight 10 estimates the same model as all of them
    X, y, offset = simulated(seed=1)
    full = logistic.fit(X, y, np.ones(len(y)), offset, penalty=1.0)
    rng = np.random.default_rng(2)
    kept = (y == 1) | (rng.random(len(y)) < 0.1)
    w = np.where(y == 1, 1.0, np.where(kept, 10.0, 0.0))
    sampled = logistic.fit(X, y, w, offset, penalty=1.0)
    assert sampled.intercept == pytest.approx(full.intercept, abs=0.3)
    assert np.allclose(sampled.coef, full.coef, atol=0.3)


def test_contributions_add_up_to_the_log_odds_and_the_saved_model_scores_alike():
    X, y, offset = simulated(n=5_000)
    model = logistic.fit(X, y, np.ones(len(y)), offset, penalty=1.0)
    parts = model.contributions(X)
    assert parts.shape == (len(X), 3 + 1)
    assert np.allclose(parts.sum(axis=1), model.raw(X), atol=1e-5)  # scoring runs in float32
    again = logistic.Logistic.from_dict(model.to_dict())
    assert np.allclose(again.raw(X), model.raw(X))
    assert list(model.importance().index) == list(X.columns)


def test_out_of_fold_picks_a_penalty_and_scores_every_row():
    from daisi_team_master_corgis_floodsense.ml import gated, ordinal
    from tests.test_ml_gated import ROWS, SPLIT

    dev = gated.development_rows(ROWS, SPLIT)
    X = ordinal.design(dev, ordinal.DEFAULTS.inputs)
    y = dev["alert_label"].to_numpy()
    scores, penalty = logistic.out_of_fold(
        X, y, np.ones(len(y)), dev["ts"], dev["area"], [{}] * len(SPLIT.folds), SPLIT
    )
    assert penalty in logistic.PENALTIES
    assert np.isfinite(scores).all() and ((scores > 0) & (scores < 1)).all()
    # out of fold, the alert rows still rank above the rest
    assert scores[y == 1].mean() > 5 * scores[y == 0].mean()
