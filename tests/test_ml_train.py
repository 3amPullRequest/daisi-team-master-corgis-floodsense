"""The parts of the training job that need neither Spark nor MLflow."""

import math

import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import gated, train


def coverage(test_known: float, test_blind: float) -> pd.DataFrame:
    """train.load_coverage's shape: days per split by label and feature window."""
    rows = [
        ("train", C.RECORDED, C.RECORDED, None, 763.3),
        ("train", C.RECORDED, C.MISSING, C.FORECAST_GAP, 209.9),
        # missing labels count for neither side
        ("train", C.MISSING, C.RECORDED, None, 290.2),
        ("test", C.RECORDED, C.RECORDED, None, test_known),
        ("test", C.RECORDED, C.MISSING, C.RAIN_GAP, test_blind),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "split",
            "label_window",
            "feature_window",
            "feature_missing_reason",
            "days",
        ],
    )


def test_usable_share_is_of_the_recorded_label_time():
    c = coverage(290.0, 5.3)
    assert train.usable_share(c, "train") == pytest.approx(763.3 / (763.3 + 209.9))
    assert train.usable_share(c, "test") == pytest.approx(290.0 / 295.3)
    assert train.usable_share(c, "embargo") == 0.0


def test_the_test_is_refused_until_its_features_land():
    with pytest.raises(SystemExit, match="features for 0% of its recorded label time"):
        train.check_test_coverage(coverage(0.05, 295.2))
    train.check_test_coverage(coverage(290.0, 5.3))  # 98%: scored


def test_input_coverage_shows_an_optional_feature_that_hasnt_landed():
    # four rows, two with an alert after them; the weather has landed for the first only
    rows = pd.DataFrame(
        {
            **{c: [1.0] * 4 for c in C.MODEL_FEATURES},
            "alert_label": [1, 0, 1, 0],
        }
    )
    rows.loc[1:, "air_temp_c"] = None
    inputs = train.input_coverage(rows).set_index("feature")
    assert list(inputs.index) == list(C.MODEL_FEATURES)
    assert inputs.loc["air_temp_c", "known_share"] == 0.25
    assert inputs.loc["air_temp_c", "known_share_alert_rows"] == 0.5
    assert not inputs.loc["air_temp_c", "required"]
    assert inputs.loc["rain_15min_mm", "required"]
    assert inputs.loc["rain_15min_mm", "known_share"] == 1.0


def test_fits_rank_by_events_flagged_then_lead_then_fewer_false_alarms():
    def m(flagged, lead, false_alarms):
        return {
            "alert_flagged_before_pub": flagged,
            "alert_mean_lead_min": lead,
            "false_alarms_per_area_month": false_alarms,
        }

    assert train._rank(m(0.21, 5.1, 0.87)) > train._rank(m(0.20, 5.4, 0.86))
    assert train._rank(m(0.21, 5.1, 0.80)) > train._rank(m(0.21, 5.1, 0.87))
    assert train._rank(m(math.nan, 0, 0)) < train._rank(m(0.0, 0, 0.5))


def test_a_flood_flagged_counts_flood_weight_times_in_the_rank():
    def m(alert_share, flood_share):
        # 20 PUB events, 2 of them flash floods
        return {
            "alert_events": 20,
            "alert_flagged_before_pub": alert_share,
            "flood_events": 2,
            "flood_flagged_before_pub": flood_share,
            "alert_mean_lead_min": 5.0,
            "false_alarms_per_area_month": 0.8,
        }

    # one more warning flagged against one more flood flagged
    warnings, floods = m(0.30, 0.0), m(0.25, 0.5)
    assert train._rank(warnings) > train._rank(floods)  # unweighted, as before 21 Oct
    assert train._rank(floods, 10) > train._rank(warnings, 10)


def test_job_parameters_parse():
    args = train._parser().parse_args(
        [
            "--catalog",
            "flood_dev",
            "--experiment",
            "/Users/x/floodsense",
            "--models",
            "gated:gate_recall=0.85,gated:gate_recall=0.9",
            "--score-test",
            "false",
        ]
    )
    assert [s.settings().gate_recall for s in args.models] == [0.85, 0.9]
    assert args.score_test is False
    args = train._parser().parse_args(
        ["--catalog", "c", "--experiment", "e", "--score-test", "true"]
    )
    assert args.score_test is True and args.model == C.REGISTERED_MODEL
    # the gated pair by default (it won on 23 Oct), with the rule as its override if it earns its place
    assert [s.name for s in args.models] == ["gated"] and args.override is True
    assert args.compare_without == ["tide"]  # the job's default
    args = train._parser().parse_args(
        ["--catalog", "c", "--experiment", "e", "--compare-without", "antecedent, forecast"]
    )
    assert args.compare_without == ["antecedent", "forecast"]
    assert set(args.compare_without) <= set(C.FEATURE_GROUPS)
    args = train._parser().parse_args(
        ["--catalog", "c", "--experiment", "e", "--compare-without", ""]
    )
    assert args.compare_without == []


def test_model_specs_parse_into_settings():
    base = ["--catalog", "c", "--experiment", "e"]
    args = train._parser().parse_args(
        [*base, "--models", "ordinal, ordinal:flood=logistic:flood_weight=3, ordinal:extended, gated, gated:extended"]
        + ["--register", "false", "--override", "false"]
    )
    assert args.register is False and args.override is False
    ordinal, logistic, extended, pair, extended_pair = args.models
    s = logistic.settings(max_false_alarms=0.87, flood_weight=1.0, flood_value=10.0)
    # a spec's own options win over the run's defaults
    assert (s.alert_learner, s.flood_learner, s.flood_weight, s.max_false_alarms) == ("lightgbm", "logistic", 3.0, 0.87)
    assert ordinal.settings(flood_weight=2.0).flood_weight == 2.0
    # the feature sets: base (v5's) by default, extended with the dew point for the humidity and the prior rain
    assert ordinal.settings().inputs == ("hour_sgt", *C.BASE_FEATURES)
    assert pair.settings().inputs == ("hour_sgt", *C.BASE_FEATURES)  # the pair as v5 reads
    assert extended.settings().inputs == ("hour_sgt", *C.EXTENDED_FEATURES)
    assert extended_pair.settings().inputs == ("hour_sgt", *C.EXTENDED_FEATURES)
    assert (pair.prefix, ordinal.prefix) == ("pair", "model")
    assert pair.params(pair.settings())["gate_recall"] == 0.9
    # drop= leaves feature groups out, on top of the feature set
    no_prior = train._parser().parse_args([*base, "--models", "gated:extended:drop=antecedent+tide"]).models[0].settings()
    assert not set(C.PRIOR_RAIN) & set(no_prior.inputs) and "tide_height_m" not in no_prior.inputs
    assert "dew_point_c" in no_prior.inputs and "rel_humidity_pct" not in no_prior.inputs
    # the alert rung may be the gated pair, the flood rung not
    gated_alert = train._parser().parse_args([*base, "--models", "ordinal:alert=gated:gate_recall=0.85"]).models[0]
    assert (gated_alert.settings().alert_learner, gated_alert.settings().gate_recall) == ("gated", 0.85)
    for bad in ("forest", "ordinal:area_effect=category", "gated:alert=logistic", "ordinal:flood=gated", "gated:drop=hail", ""):
        with pytest.raises(SystemExit):
            train._parser().parse_args([*base, "--models", bad])


def test_specs_add_feature_groups_and_a_prior_half_life():
    base = ["--catalog", "c", "--experiment", "e"]
    pair, both, hybrid = train._parser().parse_args(
        [*base, "--models", "gated:add=pub_alerts, gated:add=pub_alerts:half_life=730, ordinal:alert=gated:add=pub_alerts"]
    ).models
    # add= reads a group beyond the set, in gold's order
    read = ("hour_sgt", *(c for c in C.MODEL_FEATURES if c in C.BASE_FEATURES or c in C.PUB_ALERTS))
    assert pair.settings().inputs == read and hybrid.settings().inputs == read
    assert set(C.PUB_ALERTS) <= set(hybrid.settings().pair_settings().inputs)
    # half_life= weighs the pair's credibility prior by recency; v5 and the default weigh every day alike
    s = both.settings()
    assert s.prior_half_life_days == 730.0 and both.params(s)["prior_half_life_days"] == 730.0
    assert pair.settings().prior_half_life_days is None
    for bad in ("gated:add=hail", "gated:half_life=0", "ordinal:half_life=365"):
        with pytest.raises(SystemExit):
            train._parser().parse_args([*base, "--models", bad])


def test_specs_swap_single_features_and_refit_with_another_seed():
    base = ["--catalog", "c", "--experiment", "e"]
    grouped, storm, again, hybrid = train._parser().parse_args(
        [
            *base,
            "--models",
            "gated:drop=forecast_rain_class:add=forecast_rain_group4, gated:add=storm+surplus, gated:seed=2, "
            "ordinal:alert=gated:seed=3:add=storm",
        ]
    ).models
    # one feature swapped for another, in gold's order: the forecast's class for its four groups
    inputs = grouped.settings().inputs
    assert "forecast_rain_class" not in inputs and "forecast_rain_group4" in inputs
    assert inputs == ("hour_sgt", *(c for c in C.MODEL_FEATURES if c in {*C.BASE_FEATURES, "forecast_rain_group4"}
                                    and c != "forecast_rain_class"))  # fmt: skip
    # groups and features mix, and the storm so far reads the age and every drain rate
    assert set(C.STORM_HISTORY) <= set(storm.settings().inputs)
    assert not set(C.FORECAST_GROUPED) & set(storm.settings().inputs)
    # seed= refits the same model with another sample of the quiet rows, for either family
    s = again.settings()
    assert (s.seed, again.params(s)["seed"]) == (2, 2) and s.inputs == ("hour_sgt", *C.BASE_FEATURES)
    assert storm.settings().seed == 0 and hybrid.settings().seed == 3
    assert C.STORM_AGE in hybrid.settings().pair_settings().inputs
    for bad in ("gated:add=rain_surplus_7mmh_mm", "gated:drop=area", "gated:seed=one"):
        with pytest.raises(SystemExit):
            train._parser().parse_args([*base, "--models", bad])


def test_specs_add_the_ground_from_the_national_map():
    base = ["--catalog", "c", "--experiment", "e"]
    both, lines, hybrid = train._parser().parse_args(
        [*base, "--models", "gated:add=storm+surplus+map_line+map_polygon, gated:add=map_line, "
         "ordinal:alert=gated:add=map_polygon"]
    ).models  # fmt: skip
    # v10's features and the ground beside them, in gold's order: the ground is the last of gold's columns
    read = {*C.BASE_FEATURES, *C.STORM_HISTORY, *C.MAP_FEATURES}
    assert both.settings().inputs == ("hour_sgt", *(c for c in C.MODEL_FEATURES if c in read))
    assert both.settings().inputs[-len(C.MAP_FEATURES):] == C.MAP_FEATURES
    assert set(C.MAP_LINE) <= set(lines.settings().inputs) and not set(C.MAP_POLYGON) & set(lines.settings().inputs)
    assert set(C.MAP_POLYGON) <= set(hybrid.settings().pair_settings().inputs)
    # the default reads none of it: v5's and v10's features stay as they were
    assert not set(C.MAP_FEATURES) & set(gated.DEFAULTS.inputs)


def test_specs_add_the_drains_response():
    base = ["--catalog", "c", "--experiment", "e"]
    drains, rises, instead = train._parser().parse_args(
        [*base, "--models", "gated:add=storm+surplus+drain, gated:add=storm+surplus+drain+rise, "
         "gated:add=storm+drain+rise"]
    ).models  # fmt: skip
    # v10's features with the drains' flows and overflows beside them, then each flow's rise too, then both for the surplus
    assert set(C.DRAIN_STATE) <= set(drains.settings().inputs) and not set(C.DRAIN_RISE) & set(drains.settings().inputs)
    assert set(C.DRAIN_RESPONSE) | set(C.STORM_HISTORY) <= set(rises.settings().inputs)
    assert set(C.DRAIN_RESPONSE) <= set(instead.settings().inputs)
    assert not set(C.RAIN_SURPLUS) & set(instead.settings().inputs) and C.STORM_AGE in instead.settings().inputs
    # in gold's order, and the default reads none of it
    for spec in (drains, rises, instead):
        inputs = spec.settings().inputs
        assert inputs == ("hour_sgt", *(c for c in C.MODEL_FEATURES if c in inputs))
    assert not set(C.DRAIN_RESPONSE) & set(gated.DEFAULTS.inputs)


def test_specs_fix_the_tiers_as_chances():
    base = ["--catalog", "c", "--experiment", "e"]
    live, default = train._parser().parse_args(
        [*base, "--models", "gated:add=storm+surplus:medium=0.01:high=0.05, gated"]
    ).models
    s = live.settings()
    assert (s.medium_chance, s.high_chance) == (0.01, 0.05)
    assert (live.params(s)["medium_chance"], live.params(s)["high_chance"]) == (0.01, 0.05)
    assert default.settings().medium_chance is None and default.settings().high_chance is None
    for bad in ("gated:medium=0.05:high=0.01", "gated:medium=2", "ordinal:medium=0.01"):
        with pytest.raises(SystemExit):
            train._parser().parse_args([*base, "--models", bad])


def test_high_is_judged_alone_as_medium_is():
    preds = pd.DataFrame(
        {"area": ["A", "A", "B"], "ts": pd.to_datetime(["2024-01-01"] * 3), "tier": ["Medium", "High", "High"], "p_alert": [0.02, 0.06, 0.07]}
    )
    alone = train._high_only(preds)
    assert alone["tier"].tolist() == ["Medium", "Medium"] and alone["p_alert"].tolist() == [0.06, 0.07]
