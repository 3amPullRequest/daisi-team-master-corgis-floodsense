"""diagnostics.py on the synthetic storms: every count adds up, and agrees with the fit's stats and its scoreboard."""

import math

import numpy as np
import pytest

from daisi_team_master_corgis_floodsense.ml import diagnostics, gated
from daisi_team_master_corgis_floodsense.ml.scoreboard import score
from tests.test_ml_gated import ALERTS, FAST, ROWS, SPLIT


@pytest.fixture(scope="module")
def fit():
    _, fitted = gated.fit(ROWS, ALERTS, SPLIT, settings=FAST)
    board = score(fitted.predictions(), ALERTS, start=SPLIT.start, end=SPLIT.dev_end)
    return fitted, board, diagnostics.summary(fitted, board)


def test_a_row_confusion_counts_every_row_once():
    y = np.array([1, 1, 0, 0, 0, 1])
    warned = np.array([True, False, True, False, False, True])
    out = diagnostics.rows_confusion(y, warned, "x")
    assert (out["x_tp"], out["x_fp"], out["x_fn"], out["x_tn"]) == (2, 1, 1, 2)
    assert (out["x_fpr"], out["x_fnr"], out["x_precision"]) == pytest.approx((1 / 3, 1 / 3, 2 / 3))
    # nothing warned: no precision to speak of, and every positive missed
    none = diagnostics.rows_confusion(y, np.zeros(6, dtype=bool), "x")
    assert math.isnan(none["x_precision"]) and none["x_fnr"] == 1.0


def test_the_rows_agree_with_the_fit(fit):
    fitted, _, d = fit
    n = len(fitted.rows)
    assert d["medium_alert_tp"] + d["medium_alert_fp"] + d["medium_alert_fn"] + d["medium_alert_tn"] == n
    assert (d["medium_alert_tp"] + d["medium_alert_fp"]) / n == pytest.approx(fitted.stats["warned_row_share"])
    assert 1 - d["medium_alert_fnr"] == pytest.approx(fitted.stats["alert_row_recall"])
    assert 1 - d["medium_flood_fnr"] == pytest.approx(fitted.stats["flood_row_recall"])
    # High is a higher cut-off on the same chance: never more rows than Medium
    assert d["high_alert_tp"] + d["high_alert_fp"] <= d["medium_alert_tp"] + d["medium_alert_fp"]


def test_the_warnings_and_events_agree_with_the_scoreboard(fit):
    _, board, d = fit
    m = board.metrics(ci=False)
    assert d["medium_episodes"] == m["warnings"] and d["high_episodes"] == m["high_warnings"]
    assert d["medium_false_alarms"] / d["area_months"] == pytest.approx(m["false_alarms_per_area_month"])
    if m["high_warnings"]:
        assert d["high_episode_hits"] / d["high_episodes"] == pytest.approx(m["high_precision"])
    for kind in ("alert", "flood"):
        events = m[f"{kind}_events"]
        parts = d[f"{kind}_events_before_pub"] + d[f"{kind}_events_late"] + d[f"{kind}_events_missed"]
        assert parts == events
        assert d[f"{kind}_events_before_pub"] == round(m[f"{kind}_flagged_before_pub"] * events)
        assert d[f"{kind}_events_high_before_pub"] <= d[f"{kind}_events_before_pub"]
    # an alert event is timed at PUB's first alert, so it can't be warned late
    assert d["alert_events_late"] == 0


def test_the_chances_have_a_shape_a_rank_and_a_calibration(fit):
    fitted, _, d = fit
    quantiles = [d[f"p_{name}"] for name in diagnostics.QUANTILES]
    assert quantiles == sorted(quantiles) and quantiles[-1] <= d["p_max"]
    # most rows sit at the screened-out rows' chance, a few far above it: a long right tail
    assert d["p_skewness"] > 0 and d["p_mean_alert_rows"] > d["p_mean_quiet_rows"]
    assert 0.5 < d["alert_roc_auc"] <= 1
    # isotonic calibration on the out-of-fold scores keeps the expected alerts near those that came
    assert d["expected_over_observed"] == pytest.approx(1.0, abs=0.25)
    assert 0 <= d["passed_ece"] < 0.1 and d["passed_share"] == pytest.approx(fitted.passed.mean())
    budgets = [d[f"flagged_at_{fa:g}_false_alarms"] for fa in diagnostics.BUDGETS]
    assert all(np.isnan(a) or a <= b for a, b in zip(budgets, budgets[1:]))


def test_chances_without_a_gate_have_no_passed_rows():
    # an ordinal fit without the gated alert rung passes no rows on: its chances are summarised without them
    p = np.linspace(0.001, 0.2, 200)
    out = diagnostics.chances(p, (p > 0.15).astype("int8"), None)
    assert out["alert_roc_auc"] == 1.0 and not any(k.startswith("passed_") for k in out)


def test_the_tier_table_scores_other_tiers_as_the_fit_scores_its_own(fit):
    fitted, board, d = fit
    medium_p = fitted.stats["medium_p"]
    rain = ROWS.loc[fitted.rows.index, "rain_15min_mm"].to_numpy()
    table = diagnostics.tier_table(
        fitted,
        ALERTS,
        SPLIT,
        medium_p=medium_p,
        high_where={"rain_15min_mm >= 5": rain >= 5},
        candidates=FAST.cut_off_candidates,
    )
    assert set(table["definition"]) == {"medium_at", "high_at", "high_where"}
    # Medium at the fit's own cut-off (one of the same candidates) is the fit's own scoreboard
    own = table[(table["definition"] == "medium_at") & np.isclose(table["cut_off"], medium_p)]
    assert len(own) == 1
    assert own["events_before_pub"].iloc[0] == d["alert_events_before_pub"]
    assert own["medium_warnings"].iloc[0] == d["medium_episodes"]
    # the round chances are tried too, and a lower Medium never warns fewer rows
    assert {0.01, 0.05} <= set(table["cut_off"].round(6))
    medium = table[table["definition"] == "medium_at"].sort_values("cut_off")
    assert (np.diff(medium["rows_warned"]) <= 0).all()
    # High sits on the fit's Medium: never more High warnings than Medium ones, and what followed them is a subset
    high = table[table["definition"] != "medium_at"]
    assert (high["high_warnings"] <= high["medium_warnings"]).all()
    assert (high["high_warnings_flood_followed"] <= high["high_warnings_alert_followed"]).all()
    assert (high["floods_high_before_pub"] <= high["floods_before_pub"]).all()
    assert (table[table["definition"] == "high_at"]["cut_off"] > medium_p).all()


def test_the_share_of_warnings_followed_by_an_alert(fit):
    from daisi_team_master_corgis_floodsense.ml import train

    _, board, d = fit
    assert train._followed_share(board) == pytest.approx(d["medium_episode_hits"] / d["medium_episodes"])
