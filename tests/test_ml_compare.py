"""compare.py and Settings.drop: the pair fitted without the tide on the synthetic storms, where the tide is noise."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import compare, gated, train
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


def test_the_rank_and_the_rows_are_paired_as_the_sweep_ranks_and_scores(both):
    _, with_fit, with_board = both["with"]
    _, without_fit, without_board = both["without"]
    # both fits are judged on the same events, so the paired differences are the differences of their metrics
    m_with, m_without = with_board.metrics(ci=False), without_board.metrics(ci=False)
    rank = compare.paired_rank(with_board, without_board, flood_value=10.0)
    assert rank["rank_difference"] == pytest.approx(
        gated.weighted_flagged(m_with, 10.0) - gated.weighted_flagged(m_without, 10.0)
    )
    assert rank["rank_difference_lo"] <= rank["rank_difference"] <= rank["rank_difference_hi"]
    paired = compare.paired_events(with_board, without_board)
    assert paired["alert_flagged_difference"] == pytest.approx(
        m_with["alert_flagged_before_pub"] - m_without["alert_flagged_before_pub"]
    )
    assert 0 <= paired["alert_flagged_difference_p"] <= 1
    # a fit against itself: no difference anywhere, and nothing to reject
    same = compare.paired_rank(with_board, with_board, flood_value=10.0)
    assert (same["rank_difference"], same["rank_difference_p"]) == (0.0, 1.0)
    rows = with_fit.rows
    y = rows["alert_label"].to_numpy()
    itself = compare.paired_rows(with_fit.p_alert, with_fit.p_alert, y, rows["ts"])
    assert itself["log_loss_difference"] == 0.0 and itself["rows_paired"] == len(rows)
    # chances nudged towards the labels lose less, every day
    sharper = np.clip(np.where(y == 1, with_fit.p_alert + 0.05, with_fit.p_alert * 0.5), 0, 1)
    better = compare.paired_rows(sharper, with_fit.p_alert, y, rows["ts"])
    assert better["log_loss_difference_hi"] < 0 and better["log_loss_difference_p"] == 0.0
    with pytest.raises(ValueError, match="rows differ"):
        compare.paired_rows(with_fit.p_alert[1:], with_fit.p_alert, y, rows["ts"])


def test_the_sweep_pairs_each_fit_with_its_first(both):
    _, with_fit, with_board = both["with"]
    _, without_fit, without_board = both["without"]
    first = train.Reference.of(without_fit, without_board)
    versus = train.versus_first(first, with_fit, with_board, 10.0)
    assert versus["rank_difference"] == pytest.approx(
        compare.paired_rank(with_board, without_board, 10.0)["rank_difference"]
    )
    assert versus["rows_paired"] == len(with_fit.rows)
    assert versus["alert_pr_auc_difference"] == pytest.approx(
        with_fit.stats["alert_pr_auc"] - without_fit.stats["alert_pr_auc"]
    )
    assert {"alert_mcnemar_p", "flood_mcnemar_p", "log_loss_difference_p"} <= set(versus)
    # a fit of other rows can't be paired
    shifted = dataclasses.replace(with_fit, rows=with_fit.rows.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="can't be paired"):
        train.versus_first(first, shifted, with_board, 10.0)


def test_a_day_bootstrap_of_differences_brackets_their_mean():
    days = pd.Series(pd.date_range("2024-01-01", periods=40, freq="D").repeat(3))
    lifted = compare.day_bootstrap_difference(pd.Series(np.full(len(days), 0.25)), days)
    assert lifted == {"": 0.25, "_lo": 0.25, "_hi": 0.25, "_p": 0.0}
    # half the days one way, half the other: no difference to speak of
    signs = pd.Series(np.where(np.arange(len(days)) // 3 % 2 == 0, 1.0, -1.0))
    even = compare.day_bootstrap_difference(signs, days)
    assert even[""] == 0.0 and even["_lo"] < 0 < even["_hi"] and even["_p"] > 0.5
    # weights count an event as many times
    weighted = compare.day_bootstrap_difference(
        pd.Series([1.0, 0.0]), pd.Series(days[:2].to_numpy()), pd.Series([9.0, 1.0])
    )
    assert weighted[""] == pytest.approx(0.9)
    assert math.isnan(compare.day_bootstrap_difference(pd.Series([], dtype=float), pd.Series([]))[""])


def test_metric_table_says_which_side_is_better():
    table = compare.metric_table(
        {"alert_flagged_before_pub": 0.3, "false_alarms_per_area_month": 0.9},
        {"alert_flagged_before_pub": 0.2, "false_alarms_per_area_month": 0.8},
    ).set_index("metric")
    assert table.loc["alert_flagged_before_pub", "better"] == "with"
    assert table.loc["false_alarms_per_area_month", "better"] == "without"
