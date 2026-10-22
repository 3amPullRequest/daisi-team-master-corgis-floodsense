"""Train the gated LightGBM pair and the rainfall rule on gold, log both to MLflow, and register the pair in Unity
Catalog. The flood-train job runs it on serverless after refreshing gold (resources/train.job.yml):

    databricks bundle run train -t dev -p flood_dev
    databricks bundle run train -t dev -p flood_dev --params score_test=true   # the single look at the test

What it reads, every value point-in-time and none of it from a missing interval:
- gold.model_rows, split 'train': the area-ticks whose label window and feature window are both recorded;
- gold.label_ticks and gold.feature_ticks: which ticks' labels or features are missing, for the scoreboard;
- silver.area_alerts: PUB's alerts, which the scoreboard judges against.

What it logs to one MLflow run: the split and its folds; how much of each split the models can use; the rule's fit;
the pair's fit (one nested run per gate recall and variant tried, the scoreboard picking one); both scored by
scoreboard.score over the development period, the pair out of fold; and whether the pair beats the rule: more lead time
at no more false alarms (its Medium cut-off is held to the rule's false-alarm rate). The pair is registered as
@challenger, and as @champion as well when it beats the rule; --register false logs it without registering, for
experiments.

The sweep (--gate-recall, --variants): every gate recall is tried with the pair as --area-effect and --flood-weight
set it, and with each extra variant ("area_effect:flood_weight", e.g. "category:1" for the pair as it was before 21
Oct). Each fit is a nested run, and `sweep.json` sets them side by side: the gate's pass share and recalls, the cut-offs,
the out-of-fold events flagged (a flood counting --flood-value times, as `_rank` ranks the fits and each fit chooses its
Medium cut-off), lead and false alarms, and the best each fit flags at 2 and 4 false alarms per area-month, which shows
how much the false-alarm budget alone holds the pair back. --flood-weight is how much a flood row weighs in training
(1 by default: 10 cost 12 points of events and caught no more floods, 22 Oct); --flood-value what a flood flagged is
worth in choosing (scoreboard.FLOOD_EVENT_VALUE).

With --compare-without (default 'tide'), it refits the chosen pair without each named group of
contract.FEATURE_GROUPS, in a nested run per group, and logs what the group adds: the headline metrics side by side, the
PUB events matched one to one, and the group's share of the models' gain (compare.py).

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
4. Fit the pair, once for each gate recall and variant asked for, each a nested MLflow run scored out of fold by the
   same scoreboard as the rule. The fit that flags the most events before PUB wins (a flood event counting
   --flood-value times), then the longer lead, then fewer false alarms (`_rank`).
5. Judge and register. The verdict is logged, the winner is logged as a pyfunc model (`log_model`, with serving.py and
   this package's source) and, unless --model is empty, registered. A registered pair gets the challenger alias, and the
   champion alias too if it beats the rule.
6. The test, only on request. `check_test_coverage` refuses before any fitting if too little of the test has features.
   `score_test` writes the predictions to ml.predictions before it logs a single test metric, so that if the write
   fails nobody has seen the test, and the rerun is still the first look.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, compare, gated
from . import contract as C
from .scoreboard import FLOOD_EVENT_VALUE, score
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
# looser false-alarm budgets (per area-month) at which sweep.json reports what each fit would flag
LOOSER_BUDGETS = (2.0, 4.0)
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

        # only the best fit so far is kept, so a long sweep holds no more than two fits of ten million rows at once
        best, sweep = None, []
        for effect, weight in [(args.area_effect, args.flood_weight), *args.variants]:
            for recall in args.gate_recall:
                settings = gated.Settings(
                    gate_recall=recall,
                    max_false_alarms=budget,
                    flood_weight=weight,
                    flood_value=args.flood_value,
                    area_effect=effect,
                )
                name = f"{effect}, flood x{weight:g}, gate recall {recall:g}"
                with mlflow.start_run(run_name=name, nested=True):
                    pair, fitted = gated.fit(
                        rows,
                        alerts,
                        split,
                        missing=missing,
                        blind=blind,
                        settings=settings,
                    )
                    board = score(fitted.predictions(), **judge)
                    metrics = board.metrics()
                    mlflow.log_params(
                        {"gate_recall": recall, "area_effect": effect, "flood_weight": weight}
                    )
                    _log_metrics("pair", fitted.stats)
                    _log_metrics("oof", metrics)
                    mlflow.log_table(fitted.curve, "medium_curve.json")
                    if fitted.area_prior is not None:
                        mlflow.log_table(fitted.area_prior, "area_prior.json")
                row = sweep_row(settings, fitted, metrics, args.flood_value)
                sweep.append(row)
                log.info("%s: %s", name, row)
                rank = _rank(metrics, args.flood_value)
                if best is None or rank > best[0]:
                    best = (rank, settings, pair, fitted, board, metrics)
                del pair, fitted, board
        _, chosen, pair, fitted, board, oof = best
        mlflow.log_table(pd.DataFrame(sweep), "sweep.json")

        mlflow.log_params(
            {
                "gate_recall": chosen.gate_recall,
                "area_effect": chosen.area_effect,
                "flood_weight": chosen.flood_weight,
                "flood_value": chosen.flood_value,
            }
        )
        _log_metrics("pair", fitted.stats)
        _log_metrics("oof", oof)
        mlflow.log_table(fitted.curve, "medium_curve.json")
        mlflow.log_table(fitted.reliability(), "reliability.json")
        mlflow.log_table(fitted.importance, "importance.json")
        mlflow.log_table(_dated(board.events), "oof_events.json")
        if fitted.area_prior is not None:
            mlflow.log_table(fitted.area_prior, "area_prior.json")
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

        for group in args.compare_without:
            compare_without(
                group, rows, alerts, split, judge, missing, blind, chosen,
                fitted, board, oof,
            )  # fmt: skip

        full_name = (
            f"{args.catalog}.{args.ml_schema}.{args.model}"
            if args.model and args.register
            else None
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


def compare_without(
    group,
    rows,
    alerts,
    split,
    judge,
    missing,
    blind,
    chosen,
    fitted,
    board,
    oof,
) -> pd.DataFrame:
    """Refit the pair without one feature group, with the chosen fit's settings (gate recall, false-alarm budget, area
    effect, flood weight), in a nested run, and log the comparison to the parent run (metrics prefixed
    without_<group>_)."""
    import mlflow

    drop = C.FEATURE_GROUPS[group]
    with mlflow.start_run(run_name=f"without {group}", nested=True):
        _, without = gated.fit(
            rows,
            alerts,
            split,
            missing=missing,
            blind=blind,
            settings=dataclasses.replace(chosen, drop=drop),
        )
        without_board = score(without.predictions(), **judge)
        without_oof = without_board.metrics()
        mlflow.log_params(
            {"gate_recall": chosen.gate_recall, "dropped": ",".join(drop)}
        )
        _log_metrics("pair", without.stats)
        _log_metrics("oof", without_oof)
        mlflow.log_table(without.importance, "importance.json")
    table = compare.metric_table(
        {**fitted.stats, **oof}, {**without.stats, **without_oof}
    )
    paired = compare.paired_events(board, without_board)
    share = compare.importance_share(fitted.importance, drop)
    mlflow.log_table(table, f"without_{group}_comparison.json")
    _log_metrics(f"without_{group}", {**paired, **share})
    _log_metrics(
        f"without_{group}_delta", dict(zip(table["metric"], table["difference"]))
    )
    log.info(
        "without %s: %s; paired events %s",
        group,
        table.to_dict("records"),
        paired,
    )
    return table


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


def write_predictions(
    spark, table: str, preds: pd.DataFrame, ticks: tuple | None = None
) -> None:
    """Replace these models' rows of ml.predictions (creating the table with the contract's comments if needed): all of
    them, or with `ticks` (first, last; naive SGT) only those in that span, as live scoring appends tick by tick."""
    columns = ", ".join(
        f"{c} {PREDICTIONS_TYPES[c]} COMMENT {_quoted(doc)}"
        for c, doc in C.PREDICTIONS_COLUMNS.items()
    )
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table} ({columns}) CLUSTER BY (model, ts) "
        "COMMENT 'FloodSense predictions: one row per model, NEA forecast area and tick (contract.PREDICTIONS_COLUMNS)'"
    )
    models = " OR ".join(f"model = {_quoted(m)}" for m in preds["model"].unique())
    if ticks is not None:
        first, last = (pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in ticks)
        # the session time zone is SGT (train.main, live.main), so these literals are SGT
        models = f"({models}) AND ts BETWEEN TIMESTAMP '{first}' AND TIMESTAMP '{last}'"
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
        "--flood-weight",
        type=float,
        default=gated.DEFAULTS.flood_weight,
        help="what a row a flash flood followed weighs against a warning's in training (default %(default)s)",
    )
    parser.add_argument(
        "--flood-value",
        type=float,
        default=FLOOD_EVENT_VALUE,
        help="what a flood flagged before PUB counts against another event, in ranking the fits and choosing each "
        "one's Medium cut-off (default %(default)s)",
    )
    parser.add_argument(
        "--area-effect",
        choices=gated.AREA_EFFECTS,
        default=gated.DEFAULTS.area_effect,
        help="the area as a credibility prior (the models' offset) or as a category input (default %(default)s)",
    )
    parser.add_argument(
        "--variants",
        type=_variants,
        default=[],
        help="comma-separated extra fits to compare at every gate recall, each area_effect:flood_weight (e.g. "
        "'category:1' for the pair as it was before 21 Oct); empty for none",
    )
    parser.add_argument(
        "--register",
        type=lambda text: text.strip().lower() != "false",
        default=True,
        help="'false' to log the chosen pair without registering it (experiments)",
    )
    parser.add_argument(
        "--compare-without",
        type=lambda text: [g.strip() for g in text.split(",") if g.strip()],
        default=["tide"],
        help=f"comma-separated feature groups ({', '.join(C.FEATURE_GROUPS)}) to refit the pair without and compare; "
        "empty for none",
    )
    parser.add_argument(
        "--score-test",
        type=lambda text: text.strip().lower() == "true",
        default=False,
        help="'true' to score the test split once, after fitting",
    )
    return parser


def _rank(metrics: dict[str, float], flood_weight: float = 1.0) -> tuple[float, ...]:
    """How fits are compared: events flagged before PUB (a flood event counting `flood_weight` times,
    gated.weighted_flagged), then lead, then fewer false alarms (cut-offs keep every fit within the false-alarm
    budget)."""
    return tuple(
        np.nan_to_num(x, nan=-math.inf)
        for x in (
            gated.weighted_flagged(metrics, flood_weight),
            metrics["alert_mean_lead_min"],
            -metrics["false_alarms_per_area_month"],
        )
    )


def _variants(text: str) -> list[tuple[str, float]]:
    """'category:1,credibility:10' as [(area_effect, flood_weight), ...]."""
    out = []
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        effect, _, weight = part.partition(":")
        if effect not in gated.AREA_EFFECTS or not weight:
            raise argparse.ArgumentTypeError(
                f"a variant is area_effect:flood_weight with area_effect one of {gated.AREA_EFFECTS}, not {part!r}"
            )
        out.append((effect, float(weight)))
    return out


def sweep_row(
    settings: gated.Settings, fitted: gated.Fit, metrics: dict, flood_weight: float
) -> dict[str, float]:
    """One fit of the sweep, for sweep.json: its settings, what its gate passes, its cut-offs, its out-of-fold
    scoreboard, and the best its Medium curve flags at the looser false-alarm budgets."""
    s = fitted.stats
    row = {
        "area_effect": settings.area_effect,
        "flood_weight": settings.flood_weight,
        "gate_recall": settings.gate_recall,
        **{
            k: s.get(k, math.nan)
            for k in (
                "gate_pass_share",
                "gate_alert_row_recall",
                "gate_flood_row_recall",
                "gate_alert_event_recall",
                "gate_flood_event_recall",
                "gate_rounds",
                "rater_rounds",
                "medium_p",
                "high_p",
                "brier",
                "brier_base_rate",
            )
        },
        **{
            k: metrics.get(k, math.nan)
            for k in (
                "alert_events",
                "alert_flagged_before_pub",
                "alert_mean_lead_min",
                "false_alarms_per_area_month",
                "flood_events",
                "flood_flagged_before_pub",
                "high_warnings",
                "high_precision",
            )
        },
        "weighted_flagged": gated.weighted_flagged(metrics, flood_weight),
    }
    curve = fitted.curve
    for fa in LOOSER_BUDGETS:
        within = curve[curve["false_alarms"] <= fa]
        top = (
            within.sort_values(["flagged", "lead_min"], ascending=False).iloc[0]
            if len(within)
            else None
        )
        row[f"flagged_at_{fa:g}_false_alarms"] = (
            math.nan if top is None else float(top["flagged"])
        )
        row[f"lead_min_at_{fa:g}_false_alarms"] = (
            math.nan if top is None else float(top["lead_min"])
        )
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in row.items()}


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
