"""The parts of the training job that need neither Spark nor MLflow."""

import math

import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import train


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


def test_job_parameters_parse():
    args = train._parser().parse_args(
        [
            "--catalog",
            "flood_dev",
            "--experiment",
            "/Users/x/floodsense",
            "--gate-recall",
            "0.85,0.9",
            "--score-test",
            "false",
        ]
    )
    assert args.gate_recall == [0.85, 0.9] and args.score_test is False
    args = train._parser().parse_args(
        ["--catalog", "c", "--experiment", "e", "--score-test", "true"]
    )
    assert args.score_test is True and args.model == C.REGISTERED_MODEL
    assert args.gate_recall == [0.9]
    assert args.compare_without == ["tide"]  # the job's default
    args = train._parser().parse_args(
        ["--catalog", "c", "--experiment", "e", "--compare-without", "tide, forecast"]
    )
    assert args.compare_without == ["tide", "forecast"]
    assert set(args.compare_without) <= set(C.FEATURE_GROUPS)
    args = train._parser().parse_args(
        ["--catalog", "c", "--experiment", "e", "--compare-without", ""]
    )
    assert args.compare_without == []
