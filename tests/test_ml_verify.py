"""verify.py's checks, on the pair fitted to the synthetic storms: each passes on a sound pair and fails on a broken
one. The Databricks part (loading from Unity Catalog, reading gold) runs in the flood-verify job."""

import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml import gated, verify
from tests.test_ml_gated import ALERTS, FAST, ROWS, SPLIT

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "daisi_team_master_corgis_floodsense"


@pytest.fixture(scope="module")
def pair():
    return gated.fit(ROWS, ALERTS, SPLIT, settings=FAST)[0]


@pytest.fixture(scope="module")
def rows():
    storms = ROWS[ROWS["rain_60min_mm"] > 0].head(400)
    quiet = ROWS[ROWS["rain_60min_mm"] == 0].head(200)
    return pd.concat([storms, quiet]).drop(columns=["alert_label", "flood_label", "split"]).reset_index(drop=True)


def test_a_sound_pair_passes_every_check(pair, rows):
    preds = pair.predict(rows)
    assert verify.check_outputs(pair, rows, preds)[0]
    assert verify.check_same(pair.predict(rows), preds, "again")[0]
    assert verify.check_area_prior(pair, sorted(ROWS["area"].unique()))[0]
    assert verify.check_cut_offs(pair)[0]
    ok, detail = verify.check_rain_never_lowers_risk(pair, rows)
    assert ok, detail
    assert "fell for 0" in detail
    logged = {"pair_medium_p": pair.medium_p, "pair_quiet_p": pair.quiet_p, "pair_gate_threshold": pair.gate_threshold}
    if math.isfinite(pair.high_p):
        logged["pair_high_p"] = pair.high_p
    assert verify.check_same_as_training_run(pair, logged)[0]


def test_the_saved_pair_runs_from_its_logged_code_alone(pair, rows, tmp_path):
    # the layout mlflow gives a downloaded model: artifacts/pair (GatedPair.save) and code/<package> (code_paths)
    pair.save(tmp_path / "artifacts" / "pair")
    shutil.copytree(PACKAGE, tmp_path / "code" / PACKAGE.name, ignore=shutil.ignore_patterns("__pycache__"))
    ok, detail = verify.check_logged_code_alone(tmp_path, rows, pair.predict(rows))
    assert ok, detail
    # and a broken artifact fails it
    (tmp_path / "artifacts" / "pair" / "pair.json").write_text("{}")
    assert not verify.check_logged_code_alone(tmp_path, rows, pair.predict(rows))[0]


def test_the_checks_catch_a_broken_pair(pair, rows):
    preds = pair.predict(rows)
    tampered = preds.assign(tier="Medium")
    assert not verify.check_outputs(pair, rows, tampered)[0]
    assert not verify.check_same(tampered, preds, "tampered")[0]
    broken = replace(pair, area_offsets={**pair.area_offsets, "Bedok": float("-inf")})
    assert not verify.check_area_prior(broken, sorted(ROWS["area"].unique()))[0]
    assert not verify.check_cut_offs(replace(pair, medium_p=pair.quiet_p))[0]
    assert not verify.check_same_as_training_run(pair, {"pair_medium_p": pair.medium_p * 2})[0]
    # a pair whose calibration falls with the raw score lowers the chance when it rains more
    falling = replace(pair, calibration_y=pair.calibration_y[::-1].copy())
    assert not verify.check_rain_never_lowers_risk(falling, rows)[0]


def test_an_unknown_required_feature_must_stay_unrated(pair, rows):
    blank = rows.assign(rain_15min_mm=np.nan)
    preds = pair.predict(blank)
    assert verify.check_outputs(pair, blank, preds)[0]
    rated = preds.assign(p_alert=0.5)
    assert not verify.check_outputs(pair, blank, rated)[0]


def test_flood_keys_cover_three_hours_before_pub_up_to_each_flood():
    keys = verify.flood_keys(ALERTS)
    floods_seen = ALERTS[ALERTS["kind"] == C.FLASH_FLOOD]
    assert keys["flood"].nunique() == len(floods_seen)
    first = keys.groupby("flood")["ts"].agg(["min", "max"])
    span = first["max"] - first["min"]
    # 3 hours before PUB's first alert, which comes 15 minutes before each synthetic flood, to the tick before it
    assert (span >= pd.Timedelta(hours=3)).all() and (span <= pd.Timedelta(hours=3, minutes=20)).all()


def test_splits_follow_the_contract():
    ts = pd.Series(pd.to_datetime(["2025-12-18 23:55", "2025-12-19 12:00", "2025-12-20 00:00", "2026-10-13 00:00"]))
    assert verify.split_of(ts).tolist() == ["train", "embargo", "test", "after_split"]


def test_promotion_is_off_unless_asked():
    args = verify._parser().parse_args(["--catalog", "c", "--experiment", "e"])
    assert args.promote is False and args.alias == C.CHALLENGER and args.baseline == C.CHAMPION
    assert verify._parser().parse_args(["--catalog", "c", "--experiment", "e", "--promote", "true"]).promote


def test_the_flood_report_sets_every_model_side_by_side_on_each_flood(pair):
    # as verify.main builds it: the gold rows of the ticks before each flood, the out-of-fold table as flood-train logs
    # it and as it reads back, the pair it would replace and the rule; the storms moved into the test split, so the
    # counts over floods none of them trained on have floods to count
    import json

    from daisi_team_master_corgis_floodsense.ml import floods
    from daisi_team_master_corgis_floodsense.ml.baseline import RainfallRule
    from daisi_team_master_corgis_floodsense.ml.train import _dated

    year = pd.Timedelta(days=364)  # whole weeks, so hours and weekdays keep
    rows, alerts = ROWS.assign(ts=ROWS["ts"] + year), ALERTS.assign(alert_ts=ALERTS["alert_ts"] + year)
    keys = verify.flood_keys(alerts)[["area", "ts"]].drop_duplicates()
    window = keys.merge(rows.astype({"area": object}), on=["area", "ts"], how="left").drop(
        columns=["alert_label", "flood_label", "split"]
    )
    judge = {"start": START_2026, "end": START_2026 + pd.Timedelta(days=70)}
    _, report = gated.fit(ROWS, ALERTS, SPLIT, settings=FAST)
    oof = floods.flood_table(
        report.predictions(warned_only=False).assign(ts=lambda f: f["ts"] + year), alerts, **judge
    )
    logged = json.loads(_dated(oof).to_json(orient="split", index=False))
    read_back = pd.DataFrame(logged["data"], columns=logged["columns"])
    old = gated.fit(ROWS, ALERTS, SPLIT, settings=replace(FAST, area_effect=gated.CATEGORY))[0]
    in_training = set(zip(rows["area"].astype(object), rows["ts"]))

    table, metrics = verify.flood_report(
        pair, window, alerts, judge, in_training, oof=read_back,
        others={"baseline": old, "rule": RainfallRule(15, 10.0)},
    )  # fmt: skip
    assert len(table) == (alerts["kind"] == C.FLASH_FLOOD).sum()
    assert (table["split"] == "test").all() and (table["rows_in_training"] == 12).all()
    for tag in ("oof", "baseline", "rule"):
        assert {f"lead_min_{tag}", f"tier_at_flood_{tag}", f"peak_p_alert_{tag}"} <= set(table.columns)
        assert table[f"tier_at_flood_{tag}"].notna().all()
    assert table["p_alert_at_flood_rule"].isna().all()  # the rule has no probability
    # the out-of-fold numbers survive the round trip through oof_floods.json
    assert table["lead_min_oof"].equals(oof["lead_min"]) and table["before_pub_oof"].equals(oof["before_pub"])
    assert metrics["unseen_floods"] == len(table) and metrics["unseen_baseline_floods"] == len(table)
    assert metrics["unseen_floods_warned"] == table["lead_min"].notna().sum()
    assert metrics["unseen_rule_floods_warned"] == table["lead_min_rule"].notna().sum()


START_2026 = pd.Timestamp("2026-03-01")


def test_an_area_the_pair_never_trained_on_is_named_and_scores_at_the_island_rate(pair, rows):
    ok, detail = verify.check_area_prior(pair, [*sorted(ROWS["area"].unique()), "Pulau Tekong"])
    assert ok and "never in training: Pulau Tekong" in detail
    tekong = rows.head(50).assign(area="Pulau Tekong")
    assert (pair.offsets(tekong) == 0).all()  # the island's rate
    assert pair.predict(tekong)["p_alert"].between(0, 1).all()
