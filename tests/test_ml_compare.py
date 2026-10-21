"""compare.py and Settings.drop: the pair fitted without the tide on the synthetic storms, where the tide is noise."""

import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import compare, gated
from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.scoreboard import score
from tests.test_ml_gated import ALERTS, FAST, ROWS, SPLIT

TIDE = C.FEATURE_GROUPS["tide"]


@pytest.fixture(scope="module")
def both():
    judge = {"start": SPLIT.start, "end": SPLIT.dev_end}
    out = {}
    for name, drop in (("with", ()), ("without", TIDE)):
        settings = gated.Settings(**{**FAST.__dict__, "drop": drop})
        pair, fitted = gated.fit(ROWS, ALERTS, SPLIT, settings=settings)
        out[name] = (pair, fitted, score(fitted.predictions(), ALERTS, **judge))
    return out


def test_feature_groups_cover_every_model_feature_once():
    grouped = [c for group in C.FEATURE_GROUPS.values() for c in group]
    assert sorted(grouped) == sorted(C.MODEL_FEATURES)


def test_a_pair_without_the_tide_never_reads_it(both, tmp_path):
    pair, fitted, _ = both["without"]
    assert not set(TIDE) & set(pair.inputs)
    assert set(fitted.importance["input"]) == set(pair.inputs)
    # it scores rows with or without the tide columns, and survives a save
    preds = pair.predict(ROWS.drop(columns=list(TIDE)))
    pair.save(tmp_path / "pair")
    loaded = gated.GatedPair.load(tmp_path / "pair")
    assert loaded.inputs == pair.inputs
    pd.testing.assert_frame_equal(loaded.predict(ROWS), preds)


def test_dropping_an_unknown_feature_is_refused():
    with pytest.raises(ValueError, match="not model features"):
        gated.inputs_without(("tide_height",))


def test_the_comparison_pairs_the_same_events(both):
    _, with_fit, with_board = both["with"]
    _, without_fit, without_board = both["without"]
    table = compare.metric_table(
        {**with_fit.stats, **with_board.metrics(ci=False)},
        {**without_fit.stats, **without_board.metrics(ci=False)},
    )
    assert {"alert_flagged_before_pub", "false_alarms_per_area_month", "brier"} <= set(
        table["metric"]
    )
    # NaN where a fit has no value (no High warnings: no High precision), and NaN must match NaN
    pd.testing.assert_series_equal(
        table["difference"], table["with"] - table["without"], check_names=False
    )
    paired = compare.paired_events(with_board, without_board)
    judged = with_board.events.query("kind == 'alert' and judged")
    assert paired["alert_events_paired"] == len(judged)
    flagged = paired["alert_flagged_both"] + paired["alert_flagged_only_with"]
    assert flagged == judged["before_pub"].sum()
    assert 0 <= paired["alert_mcnemar_p"] <= 1
    # the tide is noise here: it neither finds nor loses many storms
    assert (
        abs(paired["alert_flagged_only_with"] - paired["alert_flagged_only_without"])
        <= 3
    )
    share = compare.importance_share(with_fit.importance, TIDE)
    assert 0 <= share["rater_gain_share"] < 0.5


def test_metric_table_says_which_side_is_better():
    table = compare.metric_table(
        {"alert_flagged_before_pub": 0.3, "false_alarms_per_area_month": 0.9},
        {"alert_flagged_before_pub": 0.2, "false_alarms_per_area_month": 0.8},
    ).set_index("metric")
    assert table.loc["alert_flagged_before_pub", "better"] == "with"
    assert table.loc["false_alarms_per_area_month", "better"] == "without"
