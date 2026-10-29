"""The ordinal model on the synthetic storms of test_ml_gated: two rungs on every row, no gate, the warning score and
its cut-offs, the override, and the saved model as live scoring, verify.py and the app load it."""

import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import (
    floods,
    gated,
    logistic,
    models,
    ordinal,
    verify,
)
from daisi_team_master_corgis_floodsense.ml.baseline import RainfallRule
from daisi_team_master_corgis_floodsense.ml.scoreboard import score
from tests.test_ml_gated import ALERTS, FLOOD_PRONE, ROWS, SPLIT, START, TEST_START

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "daisi_team_master_corgis_floodsense"
FAST = ordinal.Settings(max_rounds=300, patience=30, cut_off_candidates=30)


@pytest.fixture(scope="module")
def fitted():
    return ordinal.fit(ROWS, ALERTS, SPLIT, settings=FAST)


@pytest.fixture(scope="module")
def linear():
    # the flood rung as a penalized logistic regression, the small-data learner
    return ordinal.fit(ROWS, ALERTS, SPLIT, settings=replace(FAST, flood_learner=ordinal.LOGISTIC))


@pytest.fixture(scope="module")
def rows():
    storms = ROWS[ROWS["rain_60min_mm"] > 0].head(400)
    quiet = ROWS[ROWS["rain_60min_mm"] == 0].head(200)
    return pd.concat([storms, quiet]).drop(columns=["alert_label", "flood_label", "split"]).reset_index(drop=True)


def test_every_alert_row_trains_both_rungs_and_the_quiet_ones_are_weighted_back_up():
    # no gate: every row an alert followed is in the sample, and the quiet rows' weights sum to their count
    dev = gated.development_rows(ROWS, SPLIT)
    alert = dev["alert_label"].to_numpy()
    w = ordinal.sample_weight(dev, alert, FAST)
    assert (w[alert == 1] == 1).all()
    storm = ordinal.stormy(dev)
    for stratum in (storm, ~storm):
        quiet = stratum & (alert == 0)
        assert w[quiet].sum() == pytest.approx(quiet.sum(), rel=0.1)
    # the stratum the gate's job went to holds the storms
    assert storm[alert == 1].mean() > 0.9 and storm.mean() < 0.5


def test_out_of_fold_tiers_flag_storms_within_the_false_alarm_budget(fitted):
    model, report = fitted
    board = score(report.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end).metrics(ci=False)
    assert board["alert_flagged_before_pub"] >= 0.5
    assert board["false_alarms_per_area_month"] <= FAST.max_false_alarms
    if math.isfinite(model.high):
        assert board["high_precision"] >= FAST.min_high_precision
    assert model.medium in set(report.curve["medium_p"])
    s = report.stats
    assert 0 < s["alert_base_rate"] < s["alert_pr_auc"] <= 1  # better than chance, threshold-free
    assert 0 < s["flood_base_rate"] < s["flood_pr_auc"] <= 1
    assert s["sampled_rows"] < s["rows"] and s["alert_row_recall"] > 0


def test_the_flood_chance_is_an_alert_chance_and_the_warning_score_counts_floods(fitted):
    model, report = fitted
    assert (report.p_flood <= report.p_alert + 1e-12).all()
    assert ((report.p_alert >= 0) & (report.p_alert <= 1)).all()
    assert np.allclose(report.score, report.p_alert + (FAST.flood_value - 1) * report.p_flood)
    chances = model.chances(ROWS)
    assert (chances["p_flood"] <= chances["p_alert"] + 1e-12).all()
    # the flood rung learns floods: its chance is higher before floods than before warnings that stayed warnings
    flood = ROWS["flood_label"] == 1
    warning_only = (ROWS["alert_label"] == 1) & ~flood
    assert chances["p_flood"][flood].mean() > chances["p_flood"][warning_only].mean()


def test_high_is_a_sub_tier_of_the_warned_rows(fitted):
    model, _ = fitted
    preds = model.predict(ROWS)
    chances = model.chances(ROWS)
    high = (preds["tier"] == "High").to_numpy()
    assert (chances["score"][high] >= model.medium).all()
    if high.any():
        assert (chances["p_flood"][high] >= model.high).all()


def test_most_warnings_land_where_floods_happen_and_dry_rows_stay_low(fitted):
    model, _ = fitted
    preds = model.predict(ROWS)
    warned = preds[preds["tier"] != "Low"]
    assert warned["area"].isin(FLOOD_PRONE).mean() >= 0.8
    quiet = ROWS["rain_60min_mm"].eq(0) & ROWS["forecast_rain_class"].eq(0) & ROWS["rain_nearby_30min_mm"].eq(0)
    assert (preds.loc[quiet, "tier"] == "Low").all()
    # but not screened out: a dry row still has a chance of its own, which no gate floors
    assert preds.loc[quiet, "p_alert"].notna().all() and preds.loc[quiet, "p_alert"].nunique() > 1


def test_predictions_are_ml_predictions_rows_with_reasons(fitted, rows):
    model, _ = fitted
    preds = model.predict(rows)
    assert list(preds.columns) == list(C.PREDICTIONS_COLUMNS)
    warned = preds["tier"] != "Low"
    assert warned.any()
    assert preds.loc[warned, "reason"].notna().all() and preds.loc[~warned, "reason"].isna().all()
    assert preds.loc[warned, "reason"].str.contains("rain|NEA forecasts|PUB has often").all()


def test_the_saved_model_scores_as_the_fitted_one_and_loads_as_either_kind(fitted, rows, tmp_path):
    model, _ = fitted
    model.save(tmp_path / "artifacts" / "model")
    again = models.load(tmp_path / "artifacts" / "model")
    assert isinstance(again, ordinal.OrdinalModel)
    assert verify.check_same(again.predict(rows), model.predict(rows), "saved")[0]
    assert isinstance(models.load_logged(tmp_path), ordinal.OrdinalModel)
    # a gated pair still loads from its own artifact, as the versions up to v5 were logged
    pair, _ = gated.fit(ROWS, ALERTS, SPLIT, settings=gated.Settings(max_rounds=100, patience=20, cut_off_candidates=20))
    pair.save(tmp_path / "old" / "artifacts" / "pair")
    assert isinstance(models.load_logged(tmp_path / "old"), gated.GatedPair)


def test_a_sound_model_passes_every_verify_check_and_a_broken_one_fails(fitted, rows, tmp_path):
    model, _ = fitted
    preds = model.predict(rows)
    assert verify.check_outputs(model, rows, preds)[0]
    assert verify.check_area_prior(model, sorted(ROWS["area"].unique()))[0]
    assert verify.check_cut_offs(model)[0]
    ok, detail = verify.check_rain_never_lowers_risk(model, rows)
    assert ok, detail
    assert "p_flood" in detail and "score" in detail
    assert verify.check_same_as_training_run(model, model.logged_cut_offs())[0]
    assert not verify.check_same_as_training_run(model, {"model_medium_cut": model.medium * 2})[0]
    assert not verify.check_cut_offs(replace(model, medium=0.0))[0]
    broken = replace(model, flood=replace(model.flood, area_offsets={**model.flood.area_offsets, "Bedok": -math.inf}))
    assert not verify.check_area_prior(broken, sorted(ROWS["area"].unique()))[0]
    # a calibration that falls with the raw score lowers the chance when it rains more
    falling = replace(
        model,
        alert=replace(model.alert, calibration={**model.alert.calibration, "y": model.alert.calibration["y"][::-1]}),
    )
    assert not verify.check_rain_never_lowers_risk(falling, rows)[0]
    # and it runs from its logged code alone, as the app loads it
    model.save(tmp_path / "artifacts" / "model")
    shutil.copytree(PACKAGE, tmp_path / "code" / PACKAGE.name, ignore=shutil.ignore_patterns("__pycache__"))
    ok, detail = verify.check_logged_code_alone(tmp_path, rows, preds)
    assert ok, detail


def test_an_unknown_required_feature_leaves_a_row_unrated(fitted, rows):
    model, _ = fitted
    blank = rows.assign(forecast_rain_class=np.nan)
    preds = model.predict(blank)
    assert (preds["tier"] == "Low").all() and preds["p_alert"].isna().all()
    assert verify.check_outputs(model, blank, preds)[0]


def test_the_flood_rung_as_a_logistic_regression(linear, rows):
    model, report = linear
    assert model.flood.learner == ordinal.LOGISTIC and model.alert.learner == ordinal.LIGHTGBM
    assert report.stats["flood_penalty"] in logistic.PENALTIES
    # the chance only rises with the rain at hand: no negative coefficient on a rising input
    lr = model.flood.model
    rising = [j for j, (c, s) in enumerate(zip(lr.columns, lr.source)) if s in gated.RISING and "_missing" not in c]
    assert rising and (lr.coef[rising] >= 0).all()
    assert verify.check_rain_never_lowers_risk(model, rows)[0]
    assert (report.p_flood <= report.p_alert + 1e-12).all()


def test_the_override_is_kept_only_if_it_flags_more_and_then_always_warns(fitted):
    model, report = fitted
    dev = gated.development_rows(ROWS, SPLIT)
    rule = RainfallRule(15, 5.0)
    chosen, refit, table = ordinal.add_override(model, report, dev, ALERTS, SPLIT, rule, settings=FAST)
    assert table["override_mm"].isna().iloc[0]  # the model alone is the first option
    alone = table.iloc[0]
    best = table[table["within_budget"]].sort_values(
        ["weighted_flagged", "lead_min"], ascending=False
    ).iloc[0]
    assert refit.stats["medium_cut"] == chosen.medium
    board = score(refit.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end).metrics(ci=False)
    assert board["false_alarms_per_area_month"] <= FAST.max_false_alarms
    assert gated.weighted_flagged(board, FAST.flood_value) >= alone["weighted_flagged"] - 1e-12
    assert gated.weighted_flagged(board, FAST.flood_value) == pytest.approx(best["weighted_flagged"])
    # an override in place warns every row whose rain passes it, with the rule's words
    forced = replace(model, override=(15, 5.0))
    preds = forced.predict(ROWS)
    heavy = (ROWS["rain_15min_mm"] >= 5.0) & ROWS[list(C.REQUIRED_FEATURES)].notna().all(axis=1)
    assert (preds.loc[heavy, "tier"] != "Low").all()
    only_rule = heavy & (forced.chances(ROWS)["score"] < forced.medium)
    if only_rule.any():
        assert preds.loc[only_rule, "reason"].str.contains("in the last 15 min").all()
    assert "R>=5mm/15min" in forced.name and forced.logged_cut_offs()["model_override_mm"] == 5.0


def test_each_flood_gets_a_tier_and_a_chance(fitted):
    model, _ = fitted
    table = floods.flood_table(model.predict(ROWS), ALERTS, start=START, end=TEST_START)
    assert len(table) == (ALERTS["kind"] == C.FLASH_FLOOD).sum()
    assert table["tier_at_flood"].isin(C.TIERS).all() and table["p_alert_at_flood"].between(0, 1).all()


def test_the_reliability_table_has_both_rungs(fitted):
    _, report = fitted
    table = report.reliability()
    assert set(table["rung"]) == {ordinal.ALERT, ordinal.FLOOD}
    assert table.groupby("rung")["rows"].sum().eq(len(report.rows)).all()


def test_the_base_features_are_what_the_gated_pair_reads_and_the_default():
    s = ordinal.Settings()
    assert s.features == ordinal.BASE and s.inputs == (ordinal.HOUR, *C.BASE_FEATURES)
    assert "rel_humidity_pct" in s.inputs and not set(C.PRIOR_RAIN) & set(s.inputs)
    extended = ordinal.Settings(features=ordinal.EXTENDED)
    assert {*C.PRIOR_RAIN, "dew_point_c"} <= set(extended.inputs) and "rel_humidity_pct" not in extended.inputs
    with pytest.raises(ValueError):
        ordinal.Settings(flood_learner="forest")
    with pytest.raises(ValueError):
        ordinal.Settings(drop=("lightning",))


@pytest.fixture(scope="module")
def hybrid():
    # the gated pair as the alert rung: its gate focuses the rater, the flood rung reads every row
    return ordinal.fit(ROWS, ALERTS, SPLIT, settings=replace(FAST, alert_learner=ordinal.GATED))


def test_the_alert_rung_can_be_the_gated_pair_without_its_veto(hybrid, rows, tmp_path):
    model, report = hybrid
    pair = model.alert.model
    assert isinstance(pair, gated.GatedPair) and model.alert.learner == ordinal.GATED
    # the alert chance is the pair's
    passed, p = pair.p_alert(rows)
    assert np.allclose(model.chances(rows)["p_alert"], p, equal_nan=True)
    assert 0 < report.stats["gate_pass_share"] < 1 and "gate_threshold" in report.stats
    # a row the gate screens out still has a flood chance of its own, and a warning score high enough warns it
    screened = np.flatnonzero(~passed & np.isfinite(p))
    assert len(screened)
    chances = model.chances(rows)
    i = screened[np.argmax(chances["score"][screened])]
    assert chances["p_alert"][i] == pytest.approx(pair.quiet_p)
    lowered = replace(model, medium=float(chances["score"][i]))
    assert lowered.predict(rows)["tier"].iloc[i] != "Low"
    # it saves, loads, passes verify's checks and runs from its logged code alone, as the other rungs do
    model.save(tmp_path / "artifacts" / "model")
    again = models.load_logged(tmp_path)
    preds = model.predict(rows)
    assert verify.check_same(again.predict(rows), preds, "saved")[0]
    assert verify.check_same_as_training_run(model, model.logged_cut_offs())[0]
    assert "model_gate_threshold" in model.logged_cut_offs()
    assert verify.check_area_prior(model, sorted(ROWS["area"].unique()))[0]
    assert verify.check_cut_offs(model)[0] and verify.check_outputs(model, rows, preds)[0]
    assert verify.check_rain_never_lowers_risk(model, rows)[0]
    shutil.copytree(PACKAGE, tmp_path / "code" / PACKAGE.name, ignore=shutil.ignore_patterns("__pycache__"))
    ok, detail = verify.check_logged_code_alone(tmp_path, rows, preds)
    assert ok, detail


def test_the_gated_alert_rung_flags_storms_within_the_budget(hybrid):
    _, report = hybrid
    board = score(report.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end).metrics(ci=False)
    assert board["alert_flagged_before_pub"] >= 0.5
    assert board["false_alarms_per_area_month"] <= FAST.max_false_alarms
    assert (report.p_flood <= report.p_alert + 1e-12).all()
    assert set(report.area_prior["rung"]) == {ordinal.ALERT, ordinal.FLOOD}
