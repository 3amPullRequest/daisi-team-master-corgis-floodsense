"""Check a registered model before it goes live, and show what it said before every real flash flood. The flood-verify
job runs it on serverless after flood-train has registered a model as @challenger (resources/verify.job.yml). Either
kind is checked (models.py): the ordinal model (from 23 Oct) or a gated pair (up to v5).

    databricks bundle run train -t dev -p flood_dev --params promote=false
    databricks bundle run verify -t dev -p flood_dev                          # checks only
    databricks bundle run verify -t dev -p flood_dev --params promote=true    # and moves @champion if they all pass

The checks, on gold.area_features rows (a random sample of the ticks after training, and the hours before every flood):

- pyfunc_matches_pair: the model as MLflow loads it from Unity Catalog (flood-live's way) gives exactly what the saved
  model gives;
- logged_code_alone: a fresh Python process with only the model's logged code on its path (the app's way, for replays)
  loads the model and gives the same again, so the artifact needs nothing from this checkout;
- same_as_training_run: the artifact's cut-offs (and a pair's gate threshold) are the ones its training run logged, so
  the registered model is the one that was evaluated;
- area_prior: every credibility prior's offsets are finite (the ordinal model has one per rung); areas a prior never
  trained on (no gauge in reach then) are named, as they score at the island's rate;
- cut_offs_ordered: for a pair, 0 <= quiet_p < medium_p < high_p, medium_p < 1; for the ordinal model, a Medium cut-off
  inside the warning score's range, a High one in (0, 1] or none, and an override threshold above 0 if any;
- rain_never_lowers_risk: 5 mm more rain in every rain column of the hour, or a wetter forecast, never lowers a row's
  chance (each of the ordinal model's: p_alert, p_flood and the warning score);
- outputs: one ml.predictions row per input with the contract's columns, a tier from contract.TIERS, a chance in [0, 1]
  wherever the required features are known and none where they aren't, and a reason exactly for the warned rows.

The floods (floods.flood_table): every flash-flood event in silver.area_alerts, scored over the 3 hours before PUB's
first alert to the flood from gold.area_features, as live scoring would have seen them (an area-tick without a feature
row is unrated, as live). One row per flood, in floods.json:

- the registered model: lead time, tier and chance at the flood, peak tier and chance in the hour before it. For a
  development flood that is in-sample if its hour was in training (`rows_in_training`), so its out-of-fold numbers
  from the training run (oof_floods.json, `*_oof`) sit beside it; test and later floods were never trained on;
- the model it would replace (`--baseline`, default the current @champion, `*_baseline`) and the rainfall rule its
  training run fitted (`*_rule`), on the same ticks.

Leads longer than the scored window (3 hours before PUB's first alert) are cut at it. Everything goes to one MLflow run,
"verify v<version>": checks.json, floods.json, and per model the flood counts over the floods it never trained on.
The run fails if any check does, and only then is @champion left where it is.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract as C
from . import floods
from .live import model_rows, rule_from_name
from .models import load_logged
from .ordinal import OrdinalModel
from .scoreboard import alert_events
from .split import EMBARGO_DAYS

log = logging.getLogger(__name__)

WINDOW_BEFORE_PUB = pd.Timedelta(hours=3)  # how far before PUB's first alert a flood's ticks are scored
SAMPLE_FRACTION = 0.005  # of the area-ticks after training: about 20,000 rows
RAIN_STEP_MM = 5.0
RAIN_COLUMNS = (*(C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES), "rain_nearby_30min_mm", "rain_upwind_30min_mm")
TICK = pd.Timedelta(minutes=C.TICK_MINUTES)
DEV_END = pd.Timestamp(C.TEST_START) - pd.Timedelta(days=EMBARGO_DAYS)


def split_of(ts: pd.Series) -> pd.Series:
    """The contract's split of each (naive SGT) time: train before the embargo, embargo, test, after_split."""
    ts = pd.to_datetime(ts)
    return pd.Series(
        np.select(
            [ts < DEV_END, ts < pd.Timestamp(C.TEST_START), ts < pd.Timestamp(C.SPLIT_END)],
            ["train", "embargo", "test"],
            "after_split",
        ),
        index=ts.index,
    )


def flood_keys(alerts: pd.DataFrame) -> pd.DataFrame:
    """(flood, area, ts) for every tick from WINDOW_BEFORE_PUB before each flood event's first alert up to the flood."""
    ev = alert_events(alerts)
    ev = ev[ev["first_flood"].notna()].reset_index(drop=True)
    keys = []
    for i, e in ev.iterrows():
        first = (e["first_alert"] - WINDOW_BEFORE_PUB).floor(TICK)
        last = floods.hour_before(e["first_flood"])[-1]
        keys += [(i, e["area"], t) for t in pd.date_range(first, last, freq=TICK)]
    return pd.DataFrame(keys, columns=["flood", "area", "ts"])


# --- the checks: each returns (passed, detail) -------------------------------------------------------------------------


def check_outputs(model, rows: pd.DataFrame, preds: pd.DataFrame) -> tuple[bool, str]:
    known = rows[[c for c in C.REQUIRED_FEATURES if c in model.features]].notna().all(axis=1).to_numpy()
    p = preds["p_alert"].to_numpy(dtype=float)
    warned = (preds["tier"] != "Low").to_numpy()
    problems = {
        "rows": len(preds) != len(rows),
        "columns": list(preds.columns) != list(C.PREDICTIONS_COLUMNS),
        "tiers": not preds["tier"].isin(C.TIERS).all(),
        "chance_where_known": not (np.isfinite(p[known]) & (p[known] >= 0) & (p[known] <= 1)).all(),
        "no_chance_where_unknown": not np.isnan(p[~known]).all(),
        "unknown_rows_low": bool(warned[~known].any()),
        "reasons": not (preds["reason"].notna().to_numpy() == warned).all(),
    }
    failed = [k for k, bad in problems.items() if bad]
    return not failed, (
        f"{len(rows)} rows, {int(known.sum())} rated, {int(warned.sum())} warned"
        + (f"; failed: {', '.join(failed)}" if failed else "")
    )


def check_same(a: pd.DataFrame, b: pd.DataFrame, what: str) -> tuple[bool, str]:
    same_tier = (a["tier"].to_numpy() == b["tier"].to_numpy()).all()
    same_p = np.allclose(a["p_alert"].to_numpy(dtype=float), b["p_alert"].to_numpy(dtype=float), equal_nan=True)
    same_reason = (a["reason"].fillna("").to_numpy() == b["reason"].fillna("").to_numpy()).all()
    ok = bool(same_tier and same_p and same_reason and len(a) == len(b))
    return ok, f"{what}: {len(a)} rows, tiers {'equal' if same_tier else 'DIFFER'}, chances " + (
        "equal" if same_p else "DIFFER"
    ) + f", reasons {'equal' if same_reason else 'DIFFER'}"


def check_logged_code_alone(root: Path, rows: pd.DataFrame, expected: pd.DataFrame) -> tuple[bool, str]:
    """The model loaded and run by a fresh interpreter that has only the model's logged code on its path (the app's
    way: ml.models where the code has it, the gated pair's own loader before 23 Oct)."""
    code = root / "code"
    with tempfile.TemporaryDirectory() as tmp:
        given, got = Path(tmp) / "rows.pkl", Path(tmp) / "preds.pkl"
        rows.to_pickle(given)
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(code)!r})\n"
            "import pandas as pd\n"
            "try:\n"
            "    from daisi_team_master_corgis_floodsense.ml import models as loader\n"
            f"    model = loader.load_logged({str(root)!r})\n"
            "except ModuleNotFoundError:\n"
            "    from daisi_team_master_corgis_floodsense.ml import gated as loader\n"
            f"    model = loader.GatedPair.load({str(root / 'artifacts' / 'pair')!r})\n"
            f"assert loader.__file__.startswith({str(code)!r}), loader.__file__\n"
            f"model.predict(pd.read_pickle({str(given)!r})).to_pickle({str(got)!r})\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp, capture_output=True, text=True, check=False
        )
        if run.returncode:
            return False, f"the logged code failed alone: {run.stderr[-800:]}"
        return check_same(pd.read_pickle(got), expected, "logged code alone vs this checkout")


def check_same_as_training_run(model, logged: dict[str, float]) -> tuple[bool, str]:
    off = {
        k: (v, logged.get(k))
        for k, v in model.logged_cut_offs().items()
        if logged.get(k) is None or not math.isclose(v, logged[k], rel_tol=1e-9)
    }
    return not off, "cut-offs as logged" if not off else f"differ from the run: {off}"


def check_area_prior(model, areas: list[str]) -> tuple[bool, str]:
    """Every offset finite. An area a prior never trained on (no development rows: no gauge in reach then) has no
    offset and scores at the island's rate (credibility.area_offsets gives it 0), as Bühlmann-Straub gives an area with
    no exposure no credibility; those are named, not failed (Pulau Tekong on 23 Oct: its one gauge reported only later)."""
    priors = model.area_priors
    if not priors:
        return False, "no area prior: this pair reads the area as a category"
    details, bad = [], []
    for rung, offsets in priors.items():
        bad += [f"{rung}: {a}" for a, v in offsets.items() if not math.isfinite(v)]
        lo, hi = min(offsets.values()), max(offsets.values())
        details.append(f"{rung} prior: {len(offsets)} areas, log relative risk {lo:.2f} to {hi:.2f}")
    unseen = sorted(set(areas) - set().union(*priors.values()))
    return not bad, (
        "; ".join(details)
        + (f"; at the island's rate, never in training: {', '.join(unseen)}" if unseen else "")
        + (f"; not finite: {sorted(bad)}" if bad else "")
    )


def check_cut_offs(model) -> tuple[bool, str]:
    if isinstance(model, OrdinalModel):
        top = 1 + (model.flood_value - 1)  # the warning score of a row with both chances at 1
        ok = 0 < model.medium < top and (math.isinf(model.high) or 0 < model.high <= 1)
        ok &= model.override is None or model.override[1] > 0
        over = f", override at {model.override[1]:g} mm in {model.override[0]} min" if model.override else ""
        return ok, f"Medium at a warning score of {model.medium:.3g} (of {top:g}), High at p_flood {model.high:.3g}{over}"
    ok = 0 <= model.quiet_p < model.medium_p < 1 and model.medium_p < model.high_p
    ok &= model.override is None or model.override[1] > 0
    over = f", override at {model.override[1]:g} mm in {model.override[0]} min" if model.override else ""
    return ok, f"quiet {model.quiet_p:.3g} < Medium {model.medium_p:.3g} < High {model.high_p:.3g}{over}"


def check_rain_never_lowers_risk(model, rows: pd.DataFrame) -> tuple[bool, str]:
    before = model.chances(rows)
    wetter = rows.copy()
    for c in RAIN_COLUMNS:
        if c in model.features:
            wetter[c] = wetter[c] + RAIN_STEP_MM  # NaN stays NaN: an unknown reading stays unknown
    wetter["forecast_rain_class"] = (wetter["forecast_rain_class"] + 1).clip(upper=max(C.RAIN_CLASSES))
    after = model.chances(wetter)
    details, fell = [], False
    for name in (k for k in before if k != "known"):
        b, a = before[name], after[name]
        rated = np.isfinite(b) & np.isfinite(a)
        lower = rated & (a < b - 1e-12)
        rose = rated & (a > b + 1e-12)
        fell |= bool(lower.any())
        details.append(
            f"{name} over {int(rated.sum())} rows: rose for {int(rose.sum())}, fell for {int(lower.sum())}"
            + (f" (worst {float((b - a)[lower].max()):.3g})" if lower.any() else "")
        )
    return not fell, "; ".join(details)


# --- on Databricks -----------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import mlflow
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from .train import _dated, _log_metrics, _missing_ticks

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")  # naive pandas times are SGT
    gold, silver = f"{args.catalog}.{args.gold_schema}", f"{args.catalog}.{args.silver_schema}"
    full = f"{args.catalog}.{args.ml_schema}.{args.model}"
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(full, args.alias).version
    mv = client.get_model_version(full, version)
    training = client.get_run(mv.run_id).data
    log.info("verifying %s v%s (@%s), trained in run %s", full, version, args.alias, mv.run_id)
    tmp = Path(tempfile.mkdtemp())
    root = Path(mlflow.artifacts.download_artifacts(f"models:/{full}/{version}", dst_path=str(tmp / "model")))
    pair = load_logged(root)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{full}@{args.alias}")

    areas = sorted(r["area"] for r in spark.read.table(f"{silver}.forecast_areas").select("area").collect())
    alerts = spark.read.table(f"{silver}.{C.AREA_ALERTS}").select("area", "alert_ts", "kind").toPandas()
    missing = _missing_ticks(spark, f"{gold}.{C.LABEL_TICKS}", "label_window")
    blind = _missing_ticks(spark, f"{gold}.{C.FEATURE_TICKS}", "feature_window")
    features = spark.read.table(f"{gold}.{C.AREA_FEATURES}").select("area", "ts", *C.MODEL_FEATURES)

    # the hours before every flood, as live scoring would have had them: no feature row is an unrated tick. Two floods'
    # windows can overlap in one area, so each area-tick is scored once.
    keys = flood_keys(alerts)
    ticks = keys[["area", "ts"]].drop_duplicates().reset_index(drop=True)
    trained = (
        spark.read.table(f"{gold}.{C.MODEL_ROWS}")
        .where(F.col("split") == "train")
        .select("area", "ts", F.lit(True).alias("in_training"))
    )
    got = (
        spark.createDataFrame(ticks)
        .join(features, ["area", "ts"], "left")
        .join(trained, ["area", "ts"], "left")
        .toPandas()
    )
    got["ts"] = pd.to_datetime(got["ts"]).astype("datetime64[ns]")
    got = got.sort_values(["area", "ts"]).reset_index(drop=True)
    in_training = set(zip(got.loc[got["in_training"].eq(True), "area"], got.loc[got["in_training"].eq(True), "ts"]))
    window = _as_rows(got)
    sample = _as_rows(
        features.where(F.col("ts") >= F.lit(C.TEST_START.isoformat()).cast("timestamp"))
        .sample(fraction=SAMPLE_FRACTION, seed=0)
        .toPandas()
    )
    rows = pd.concat([sample, window], ignore_index=True)
    log.info("%d sample rows after training, %d rows before %d floods", len(sample), len(window), keys["flood"].nunique())

    preds = pair.predict(rows)
    checks = {
        "outputs": check_outputs(pair, rows, preds),
        "pyfunc_matches_pair": check_same(pyfunc.predict(rows), preds, f"pyfunc from Unity Catalog vs {type(pair).__name__}"),
        "logged_code_alone": check_logged_code_alone(root, rows, preds),
        "same_as_training_run": check_same_as_training_run(pair, training.metrics),
        "area_prior": check_area_prior(pair, areas),
        "cut_offs_ordered": check_cut_offs(pair),
        "rain_never_lowers_risk": check_rain_never_lowers_risk(pair, rows),
    }
    for name, (ok, detail) in checks.items():
        log.info("%s %s: %s", "PASS" if ok else "FAIL", name, detail)
    passed = all(ok for ok, _ in checks.values())

    # every flood, by the registered model, the model it would replace, and the rule its training run fitted
    judge = {
        "start": pd.Timestamp(C.LABELS_START),
        "end": pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None).ceil("D"),
        "missing": missing,
        "blind": blind,
    }
    others = {}
    baseline = _baseline(mlflow, client, full, args.baseline, version, tmp)
    if baseline is not None:
        others["baseline"] = baseline[1]
    rule_name = training.params.get("rule")
    if rule_name:
        others["rule"] = rule_from_name(rule_name)
    table, metrics = flood_report(
        pair,
        window,
        alerts,
        judge,
        in_training,
        oof=_artifact_table(mlflow, mv.run_id, "oof_floods.json"),
        others=others,
    )
    metrics |= {
        "checks_passed": float(sum(ok for ok, _ in checks.values())),
        "checks": float(len(checks)),
    }

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"verify v{version}"):
        mlflow.log_params(
            {
                "model": full,
                "version": version,
                "alias": args.alias,
                "training_run": mv.run_id,
                "baseline": f"v{baseline[0]}" if baseline else "",
                "rule": rule_name or "",
            }
        )
        _log_metrics("verify", metrics)
        mlflow.log_dict({k: {"passed": ok, "detail": d} for k, (ok, d) in checks.items()}, "checks.json")
        mlflow.log_table(_dated(table), "floods.json")
        promoted = False
        beats_rule = training.metrics.get("beats_rule") == 1.0
        if args.promote and passed and beats_rule:
            client.set_registered_model_alias(full, C.CHAMPION, version)
            promoted = True
        mlflow.log_metric("promoted", float(promoted))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    log.info("floods:\n%s", table.round(4).to_string())
    log.info("metrics: %s", json.dumps(metrics))
    if promoted:
        log.info("moved @%s to v%s", C.CHAMPION, version)
    elif args.promote:
        log.warning("@%s left where it was: %s", C.CHAMPION, "a check failed" if not passed else "it doesn't beat the rule")
    return 0 if passed else 1


SHOWN = ["lead_min", "before_pub", "tier_at_flood", "p_alert_at_flood", "peak_tier", "peak_p_alert"]
KEY = ["area", "flood_ts"]


def flood_report(
    pair,
    window: pd.DataFrame,
    alerts: pd.DataFrame,
    judge: dict,
    in_training=frozenset(),
    *,
    oof: pd.DataFrame | None = None,
    others: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """floods.json and its counts. The model's floods.flood_table over `window` (gold rows of the ticks before each
    flood), with each flood's split and how many of its hour's ticks the model trained on (`in_training`, a set of
    (area, ts)); beside it the training run's out-of-fold table (`oof`, oof_floods.json read back, as `*_oof`) and the
    tables of `others` ({tag: anything with predict(rows)}, e.g. the model it would replace and the rule, as `*_<tag>`).
    The counts (floods.summary) are over the floods none of them trained on: the test split and after."""
    table = floods.flood_table(pair.predict(window), alerts, **judge)
    table.insert(4, "split", split_of(table["flood_ts"]))
    # a development flood with 0 here was outside model_rows (its label window missing): the final pair never saw its
    # hour, only, perhaps, the rest of its storm day
    table.insert(
        5,
        "rows_in_training",
        [sum((a, t) in in_training for t in floods.hour_before(f)) for a, f in zip(table["area"], table["flood_ts"])],
    )
    if oof is not None:
        oof = oof.assign(flood_ts=pd.to_datetime(oof["flood_ts"]).astype("datetime64[ns]"))
        table = table.merge(_suffixed(oof, "oof"), on=KEY, how="left")
    for tag, model in (others or {}).items():
        table = table.merge(_suffixed(floods.flood_table(model.predict(window), alerts, **judge), tag), on=KEY, how="left")

    unseen = table[table["split"].isin(["test", "after_split"])]
    metrics = floods.summary(unseen, prefix="unseen_")
    for tag in others or {}:
        mine = unseen[["area", "flood_ts", "judged", *(f"{c}_{tag}" for c in SHOWN)]]
        metrics |= floods.summary(mine.rename(columns={f"{c}_{tag}": c for c in SHOWN}), prefix=f"unseen_{tag}_")
    return table, metrics


def _suffixed(table: pd.DataFrame, tag: str) -> pd.DataFrame:
    return table[KEY + SHOWN].rename(columns={c: f"{c}_{tag}" for c in SHOWN})


def _as_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """gold.area_features rows as the scorer takes them (live.model_rows: features as float64), ts at nanoseconds."""
    rows = model_rows(frame)
    return rows.assign(ts=pd.to_datetime(rows["ts"]).astype("datetime64[ns]"))


def _artifact_table(mlflow, run_id: str, path: str) -> pd.DataFrame | None:
    try:
        local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    except Exception as exc:  # noqa: BLE001 - a run from before 23 Oct has no flood table
        log.warning("no %s in run %s: %s", path, run_id, exc)
        return None
    d = json.loads(Path(local).read_text())
    return pd.DataFrame(d["data"], columns=d["columns"])


def _baseline(mlflow, client, full: str, baseline: str, version: str, tmp: Path):
    """(version, model) of the model to compare with: an alias or a version number; None if empty or the same."""
    if not baseline:
        return None
    other = baseline if baseline.isdigit() else client.get_model_version_by_alias(full, baseline).version
    if str(other) == str(version):
        return None
    root = Path(mlflow.artifacts.download_artifacts(f"models:/{full}/{other}", dst_path=str(tmp / f"v{other}")))
    return str(other), load_logged(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ml-schema", default="ml")
    parser.add_argument("--model", default=C.REGISTERED_MODEL, help="registered model name in <catalog>.<ml schema>")
    parser.add_argument("--experiment", required=True, help="MLflow experiment path for the verify run")
    parser.add_argument("--alias", default=C.CHALLENGER, help="the version to verify (default %(default)s)")
    parser.add_argument(
        "--baseline",
        default=C.CHAMPION,
        help="the model to compare it with, an alias or a version number (default %(default)s); empty for none",
    )
    parser.add_argument(
        "--promote",
        type=lambda text: text.strip().lower() == "true",
        default=False,
        help="'true' to move @champion to the verified version if every check passes and it beats the rule",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
