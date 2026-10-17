"""Train the gated LightGBM pair and the rainfall rule on gold, log both to MLflow, and register the pair in Unity
Catalog. The flood-train job runs it on serverless after refreshing gold (resources/train.job.yml):

    databricks bundle run train -t dev -p flood_dev
    databricks bundle run train -t dev -p flood_dev --params score_test=true   # the single look at the test

What it reads, every value point-in-time and none of it from a missing interval:
- gold.model_rows, split 'train': the area-ticks whose label window and feature window are both recorded;
- gold.label_ticks and gold.feature_ticks: which ticks' labels or features are missing, for the scoreboard;
- silver.area_alerts: PUB's alerts, which the scoreboard judges against.

What it logs to one MLflow run: the split and its folds; how much of each split the models can use; the rule's fit;
the pair's fit (one nested run per gate recall tried, the scoreboard picking one); both scored by scoreboard.score over
the development period, the pair out of fold; and whether the pair beats the rule: more lead time at no more false
alarms (its Medium cut-off is held to the rule's false-alarm rate). The pair is registered as @challenger, and as
@champion as well when it beats the rule.

With --score-test true it also scores the test split, once, for both models: metrics with 95% CIs, the event-by-event
tables, and the test predictions in ml.predictions. It refuses while the test's features haven't landed.

The order of work in `main`:

1. Load. The training rows come into pandas compactly (`load_rows`: area as a category, features as float32, labels as
   int8) so ten million of them fit on a serverless driver. The alerts, the ticks with missing labels and the ticks
   with missing features come with them, and so does a coverage table of how much of each split the models can use.
2. Split. `make_split` cuts the development period into folds from the alerts the rows can actually learn from
   (`gated.usable_alerts`), so no fold is made of days without rows.
3. Fit the rule first. Its false-alarm rate sets the budget the pair's Medium cut-off may use, so that the comparison
   asks for more lead time at no more false alarms.
4. Fit the pair, once for each gate recall asked for, each a nested MLflow run scored out of fold by the same
   scoreboard as the rule. The fit that flags the most events before PUB wins, then the longer lead, then fewer false
   alarms (`_rank`).
5. Judge and register. The verdict is logged, the winner is logged as a pyfunc model (`log_model`, with serving.py and
   this package's source) and, unless --model is empty, registered. A registered pair gets the challenger alias, and the
   champion alias too if it beats the rule.
6. The test, only on request. `check_test_coverage` refuses before any fitting if too little of the test has features.
   `score_test` writes the predictions to ml.predictions before it logs a single test metric, so that if the write
   fails nobody has seen the test, and the rerun is still the first look.
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, gated
from . import contract as C
from .scoreboard import score
from .split import make_split

log = logging.getLogger(__name__)

PACKAGE = (
    Path(__file__).resolve().parents[1]
)  # logged with the model, so loading it needs no install
TICKS_PER_DAY = 24 * 60 // C.TICK_MINUTES
RULE_FLOOR_MM = min(
    baseline.THRESHOLDS_MM
)  # rows with less rain are Low under every rule candidate
MIN_TEST_COVERAGE = 0.95  # share of the test's recorded label time that must have features before it is scored
PREDICTIONS_TYPES = {
    "model": "STRING",
    "area": "STRING",
    "ts": "TIMESTAMP",
    "tier": "STRING",
    "p_alert": "DOUBLE",
    "reason": "STRING",
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import mlflow
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    # naive pandas times are SGT, as contract.py reads them
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")
    gold = f"{args.catalog}.{args.gold_schema}"
    silver = f"{args.catalog}.{args.silver_schema}"

    areas = sorted(
        r["area"]
        for r in spark.read.table(f"{silver}.forecast_areas").select("area").collect()
    )
    rows = load_rows(spark, gold, "train", areas)
    alerts = (
        spark.read.table(f"{silver}.{C.AREA_ALERTS}")
        .select("area", "alert_ts", "kind")
        .toPandas()
    )
    missing = _missing_ticks(spark, f"{gold}.{C.LABEL_TICKS}", "label_window")
    blind = _missing_ticks(spark, f"{gold}.{C.FEATURE_TICKS}", "feature_window")
    coverage = load_coverage(spark, gold)
    if args.score_test:
        check_test_coverage(coverage)  # before fitting, so a refused run costs nothing
    usable = gated.usable_alerts(alerts, rows)
    split = make_split(usable)
    log.info(
        "%d training rows, %d alert rows; %d of %d alerts usable; development %s to %s",
        len(rows),
        int(rows["alert_label"].sum()),
        len(usable),
        len(alerts),
        split.start.date(),
        split.dev_end.date(),
    )

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"gated pair {datetime.now():%Y-%m-%d %H:%M}"):
        mlflow.log_params(
            {
                "development": f"{split.start.date()} to {split.dev_end.date()}",
                "test": f"{split.test_start.date()} to {split.test_end.date()}",
                "folds": len(split.folds),
                "training_rows": len(rows),
                "alert_rows": int(rows["alert_label"].sum()),
                "usable_alerts": len(usable),
            }
        )
        mlflow.log_metrics(
            {f"usable_share_{s}": usable_share(coverage, s) for s in ("train", "test")}
        )
        mlflow.log_table(coverage, "coverage.json")
        mlflow.log_table(_dated(split.summary(usable)), "split_summary.json")

        dev = gated.development_rows(rows, split)
        inputs = input_coverage(dev)
        mlflow.log_table(inputs, "input_coverage.json")
        sparse = inputs.loc[inputs["known_share"] < 0.5, "feature"].tolist()
        if sparse:
            log.warning(
                "known for under half the development rows: %s; has their source landed?",
                ", ".join(sparse),
            )
        judge = {
            "alerts": alerts,
            "start": split.start,
            "end": split.dev_end,
            "missing": missing,
            "blind": blind,
        }

        rule, rule_curve = baseline.fit(
            dev[dev["rain_60min_mm"] >= RULE_FLOOR_MM],
            alerts,
            start=split.start,
            end=split.dev_end,
            missing=missing,
            blind=blind,
        )
        rule_metrics = score(_warned(rule.predict(dev)), **judge).metrics()
        mlflow.log_param("rule", rule.name)
        _log_metrics("rule_dev", rule_metrics)
        mlflow.log_table(rule_curve, "rule_curve.json")
        log.info("rule: %s", rule.name)
        # Round 1: the pair must beat the rule on lead time and false alarms. So its Medium cut-off may raise no more
        # false alarms than the rule does, and the verdict compares lead at that matched rate.
        budget = min(
            gated.DEFAULTS.max_false_alarms, rule_metrics["false_alarms_per_area_month"]
        )
        mlflow.log_param("false_alarm_budget", round(budget, 4))

        tried = []
        for recall in args.gate_recall:
            with mlflow.start_run(run_name=f"gate recall {recall:g}", nested=True):
                pair, fitted = gated.fit(
                    rows,
                    alerts,
                    split,
                    missing=missing,
                    blind=blind,
                    settings=gated.Settings(
                        gate_recall=recall, max_false_alarms=budget
                    ),
                )
                board = score(fitted.predictions(), **judge)
                metrics = board.metrics()
                mlflow.log_param("gate_recall", recall)
                _log_metrics("pair", fitted.stats)
                _log_metrics("oof", metrics)
                mlflow.log_table(fitted.curve, "medium_curve.json")
                log.info(
                    "gate recall %g: %s",
                    recall,
                    {
                        k: fitted.stats[k]
                        for k in ("gate_pass_share", "medium_p", "high_p")
                    },
                )
                tried.append((_rank(metrics), recall, pair, fitted, board, metrics))
        _, recall, pair, fitted, board, oof = max(tried, key=lambda t: t[0])

        mlflow.log_param("gate_recall", recall)
        _log_metrics("pair", fitted.stats)
        _log_metrics("oof", oof)
        mlflow.log_table(fitted.curve, "medium_curve.json")
        mlflow.log_table(fitted.reliability(), "reliability.json")
        mlflow.log_table(fitted.importance, "importance.json")
        mlflow.log_table(_dated(board.events), "oof_events.json")
        verdict = {
            # lead before PUB's first alert, misses counting 0, over every judged PUB event
            "beats_rule_lead": oof["alert_mean_lead_min"]
            > rule_metrics["alert_mean_lead_min"],
            "within_rule_false_alarms": oof["false_alarms_per_area_month"]
            <= rule_metrics["false_alarms_per_area_month"],
        }
        beats_rule = all(verdict.values())
        mlflow.log_metrics(
            {
                **{k: float(v) for k, v in verdict.items()},
                "beats_rule": float(beats_rule),
            }
        )
        log.info(
            "out of fold: pair flags %.1f%% of events before PUB (rule %.1f%%), lead %.1f min (rule %.1f), "
            "%.2f false alarms per area-month (rule %.2f)",
            100 * oof["alert_flagged_before_pub"],
            100 * rule_metrics["alert_flagged_before_pub"],
            oof["alert_mean_lead_min"],
            rule_metrics["alert_mean_lead_min"],
            oof["false_alarms_per_area_month"],
            rule_metrics["false_alarms_per_area_month"],
        )

        full_name = (
            f"{args.catalog}.{args.ml_schema}.{args.model}" if args.model else None
        )
        info = log_model(pair, dev, fitted, full_name)
        version = info.registered_model_version if full_name else None
        if version is not None:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(full_name, C.CHALLENGER, version)
            if beats_rule:
                client.set_registered_model_alias(full_name, C.CHAMPION, version)
            log.info(
                "registered %s version %s as @%s%s",
                full_name,
                version,
                C.CHALLENGER,
                f" and @{C.CHAMPION}" if beats_rule else "",
            )

        if args.score_test:
            name = (
                f"{args.model or C.REGISTERED_MODEL} v{version}"
                if version
                else pair.name
            )
            score_test(
                spark,
                gold,
                args,
                areas,
                coverage,
                split,
                pair,
                name,
                rule,
                alerts,
                missing,
                blind,
            )
    return 0


def load_rows(spark, gold: str, split: str, areas: list[str]) -> pd.DataFrame:
    """gold.model_rows of one split as pandas: the area as a category, features as float32 and labels as int8, so ten
    million rows fit on a serverless driver."""
    from pyspark.sql import functions as F

    code = F.create_map(*(F.lit(x) for i, a in enumerate(areas) for x in (a, i)))
    frame = (
        spark.read.table(f"{gold}.{C.MODEL_ROWS}")
        .where(F.col("split") == split)
        .select(
            F.coalesce(F.try_element_at(code, F.col("area")), F.lit(-1))
            .cast("smallint")
            .alias("area_code"),
            "ts",
            *(F.col(c).cast("float").alias(c) for c in C.MODEL_FEATURES),
            F.col("alert_label").cast("tinyint").alias("alert_label"),
            F.col("flood_label").cast("tinyint").alias("flood_label"),
        )
        .toPandas()
    )
    codes = frame.pop("area_code").astype(int)
    frame.insert(0, "area", pd.Categorical.from_codes(codes, categories=areas))
    return frame


def load_coverage(spark, gold: str) -> pd.DataFrame:
    """Days per split by label window and feature window: how much of each split the models can learn from or be
    judged on."""
    return spark.sql(
        f"""
        SELECT l.split, l.label_window, f.feature_window, f.missing_reason AS feature_missing_reason,
               count(*) / {TICKS_PER_DAY}.0 AS days
        FROM {gold}.{C.LABEL_TICKS} l JOIN {gold}.{C.FEATURE_TICKS} f USING (ts)
        GROUP BY ALL ORDER BY ALL
        """
    ).toPandas()


def input_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model feature, the share of rows where it is known, among all rows and among those an alert followed. The
    optional ones may be null (a feed outage, a backfill that hasn't landed), and one that is mostly missing would look
    unimportant to the models for want of data."""
    alert = rows["alert_label"] == 1
    return pd.DataFrame(
        {
            "feature": list(C.MODEL_FEATURES),
            "required": [c in C.REQUIRED_FEATURES for c in C.MODEL_FEATURES],
            "known_share": [float(rows[c].notna().mean()) for c in C.MODEL_FEATURES],
            "known_share_alert_rows": [
                float(rows.loc[alert, c].notna().mean()) for c in C.MODEL_FEATURES
            ],
        }
    )


def usable_share(coverage: pd.DataFrame, split: str) -> float:
    """Of a split's recorded label time, the share whose features are recorded too."""
    recorded = coverage[
        (coverage["split"] == split) & (coverage["label_window"] == C.RECORDED)
    ]
    total = recorded["days"].sum()
    known = recorded.loc[recorded["feature_window"] == C.RECORDED, "days"].sum()
    return float(known / total) if total else 0.0


def log_model(
    pair: gated.GatedPair, dev: pd.DataFrame, fitted: gated.Fit, full_name: str | None
):
    """Log the pair as a pyfunc model (with this package, so loading needs no install) and register it if named."""
    import lightgbm
    import mlflow
    from mlflow.models import infer_signature

    from .serving import GatedPairModel

    # an input example with warned rows and quiet ones, so every output column has values to type
    order = np.argsort(-fitted.p_alert)
    picked = np.concatenate([order[:3], order[-2:]])
    example = dev.iloc[picked][["area", "ts", *C.MODEL_FEATURES]].astype(
        {"area": str, **{c: "float64" for c in C.MODEL_FEATURES}}
    )
    example = example.reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "pair"
        pair.save(saved)
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=GatedPairModel(),
            artifacts={"pair": str(saved)},
            code_paths=[str(PACKAGE)],
            signature=infer_signature(example, pair.predict(example)),
            input_example=example,
            pip_requirements=[
                f"lightgbm=={lightgbm.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
            ],
            registered_model_name=full_name,
        )


def score_test(
    spark, gold, args, areas, coverage, split, pair, name, rule, alerts, missing, blind
) -> None:
    """The single look at the test: both models scored with 95% CIs, predictions written, then the metrics and event
    tables logged."""
    import mlflow

    check_test_coverage(coverage)
    test = load_rows(spark, gold, "test", areas)
    window = {
        "alerts": alerts,
        "start": split.test_start,
        "end": split.test_end,
        "missing": missing,
        "blind": blind,
    }
    pair_preds = pair.predict(test).assign(model=name)
    rule_preds = rule.predict(test)
    results = {}
    for prefix, preds in (("test", pair_preds), ("rule_test", rule_preds)):
        board = score(_warned(preds), **window)
        results[prefix] = (board.metrics(), board.events)
    # Nothing about the test is logged until its predictions are written: if the write fails, nobody has seen the
    # test, and the rerun is still the first look.
    write_predictions(
        spark,
        f"{args.catalog}.{args.ml_schema}.{C.PREDICTIONS}",
        pd.concat([pair_preds, rule_preds]),
    )
    for prefix, (metrics, events) in results.items():
        _log_metrics(prefix, metrics)
        mlflow.log_table(_dated(events), f"{prefix}_events.json")


def check_test_coverage(coverage: pd.DataFrame) -> None:
    share = usable_share(coverage, "test")
    if share < MIN_TEST_COVERAGE:
        raise SystemExit(
            f"the test split has features for {share:.0%} of its recorded label time (needs {MIN_TEST_COVERAGE:.0%}): "
            "land the API backfill through the test's end, refresh flood-etl and flood-gold, then score it. Scoring "
            "it now would spend the one look at the test on part of it."
        )


def write_predictions(spark, table: str, preds: pd.DataFrame) -> None:
    """Replace these models' rows of ml.predictions (creating the table with the contract's comments if needed)."""
    columns = ", ".join(
        f"{c} {PREDICTIONS_TYPES[c]} COMMENT {_quoted(doc)}"
        for c, doc in C.PREDICTIONS_COLUMNS.items()
    )
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table} ({columns}) CLUSTER BY (model, ts) "
        "COMMENT 'FloodSense predictions: one row per model, NEA forecast area and tick (contract.PREDICTIONS_COLUMNS)'"
    )
    models = " OR ".join(f"model = {_quoted(m)}" for m in preds["model"].unique())
    frame = preds.astype({"area": str})[list(C.PREDICTIONS_COLUMNS)]
    # the rule has no probability: NULL, not NaN
    frame["p_alert"] = (
        frame["p_alert"].astype(object).where(frame["p_alert"].notna(), None)
    )
    (
        spark.createDataFrame(frame)
        .write.mode("overwrite")
        .option("replaceWhere", models)
        .saveAsTable(table)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ml-schema", default="ml")
    parser.add_argument("--experiment", required=True, help="MLflow experiment path")
    parser.add_argument(
        "--model",
        default=C.REGISTERED_MODEL,
        help="registered model name in <catalog>.<ml schema> (default %(default)s); empty to log without registering",
    )
    parser.add_argument(
        "--gate-recall",
        type=lambda text: [float(x) for x in text.split(",") if x.strip()],
        default=[gated.DEFAULTS.gate_recall],
        help="comma-separated gate recalls to try; the out-of-fold scoreboard picks one",
    )
    parser.add_argument(
        "--score-test",
        type=lambda text: text.strip().lower() == "true",
        default=False,
        help="'true' to score the test split once, after fitting",
    )
    return parser


def _rank(metrics: dict[str, float]) -> tuple[float, ...]:
    """How fits are compared: events flagged before PUB, then lead, then fewer false alarms (cut-offs keep every fit
    within the false-alarm budget)."""
    return tuple(
        np.nan_to_num(x, nan=-math.inf)
        for x in (
            metrics["alert_flagged_before_pub"],
            metrics["alert_mean_lead_min"],
            -metrics["false_alarms_per_area_month"],
        )
    )


def _missing_ticks(spark, table: str, window_column: str) -> pd.Series:
    from pyspark.sql import functions as F

    ticks = spark.read.table(table).where(F.col(window_column) == C.MISSING)
    return ticks.select("ts").toPandas()["ts"]


def _warned(preds: pd.DataFrame) -> pd.DataFrame:
    return preds[preds["tier"] != "Low"]


def _log_metrics(prefix: str, metrics: dict) -> None:
    """MLflow refuses NaN and infinity; those go to a JSON artifact with the rest."""
    import mlflow

    clean = {
        f"{prefix}_{k}": float(v)
        for k, v in metrics.items()
        if v is not None and math.isfinite(float(v))
    }
    mlflow.log_metrics(clean)
    mlflow.log_dict(
        {
            k: (None if v is None or not math.isfinite(float(v)) else float(v))
            for k, v in metrics.items()
        },
        f"metrics/{prefix}.json",
    )


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    """Timestamps as text, which MLflow tables keep exactly."""
    return frame.apply(
        lambda col: (
            col.astype(str) if pd.api.types.is_datetime64_any_dtype(col) else col
        )
    )


def _quoted(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
