"""Live scoring: flood-live's `score` task, after each refresh of silver. It computes the newest ticks' features from
silver with gold's own definitions (features.py), scores every one of the 47 areas, and writes the rows to
ml.predictions, where the app and the dashboard read them.

    databricks bundle run live -t dev -p flood_dev   # fetch, refresh silver, score

Which ticks: from the one after the last tick this model scored up to the newest tick whose rain window has landed
(the newest reading in silver, plus READING_LAG), and never more than CATCH_UP_MINUTES back, so a missed run fills its
own gap and a long outage doesn't make one run score days. A tick is scored once; nothing landed later changes it, so
what the app showed is what scorekeeping judges.

Which model: the registered pair's @champion, else the rainfall rule from the newest flood-train run (its fitted
threshold, logged as the `rule` parameter), so the app has tiers from the first trained rule on. A pair saved before a
feature group existed still loads (GatedPair.inputs). An area with no gauge reporting in the last hour has no feature
row and is Low, as in training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re

import pandas as pd

from .. import features as FT
from ..freshness import sgt
from . import contract as C
from .baseline import RainfallRule

log = logging.getLogger(__name__)

CATCH_UP_MINUTES = 120
CATCH_UP = pd.Timedelta(minutes=CATCH_UP_MINUTES)
TICK = pd.Timedelta(minutes=C.TICK_MINUTES)
LAG = pd.Timedelta(minutes=C.READING_LAG_MINUTES)
_RULE = re.compile(r"rainfall_rule rain_(\d+)min_mm M>=([\d.]+|inf) H>=([\d.]+|inf)")


def ticks_to_score(
    now: pd.Timestamp,
    newest_reading: pd.Timestamp | None,
    last_scored: pd.Timestamp | None,
    catch_up: pd.Timedelta = CATCH_UP,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The first and last tick to score (naive SGT), or None when nothing new has landed."""
    if newest_reading is None or pd.isna(newest_reading):
        return None
    last = min(now.floor(TICK), (newest_reading + LAG).floor(TICK))
    first = last if last_scored is None or pd.isna(last_scored) else last_scored + TICK
    first = max(first, last - catch_up)
    return (first, last) if first <= last else None


def rule_from_name(name: str) -> RainfallRule:
    """The rule back from its RainfallRule.name, as flood-train logs it."""
    m = _RULE.fullmatch(name.strip())
    if not m:
        raise ValueError(f"not a rainfall rule name: {name!r}")
    window, medium, high = m.groups()
    return RainfallRule(int(window), float(medium), float(high))


def model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The scorer's input: area and ts, and every model feature as float64 (the logged signature refuses int64)."""
    return frame[["area", "ts", *C.MODEL_FEATURES]].astype(
        {"area": str, **{c: "float64" for c in C.MODEL_FEATURES}}
    )


# --- on Databricks -----------------------------------------------------------------------------------------------------


def live_model(full_name: str, short_name: str, experiment: str):
    """(name for ml.predictions, a function from feature rows to prediction rows): the @champion pair, else the newest
    trained rule."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    try:
        version = (
            mlflow.MlflowClient()
            .get_model_version_by_alias(full_name, C.CHAMPION)
            .version
        )
        model = mlflow.pyfunc.load_model(f"models:/{full_name}@{C.CHAMPION}")
        name = f"{short_name} v{version}"
        log.info("scoring with %s (@%s)", name, C.CHAMPION)
        return name, lambda rows: model.predict(model_rows(rows)).assign(model=name)
    # no champion yet, or it doesn't load (any reason): fall back to the rule, loudly
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "no usable @%s for %s (%s); using the rule", C.CHAMPION, full_name, exc
        )
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="params.rule != ''",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(
            f"nothing to score with: no @{C.CHAMPION} for {full_name} and no flood-train run in {experiment}. "
            "Run flood-train once."
        )
    rule = rule_from_name(runs.iloc[0]["params.rule"])
    log.info("scoring with %s", rule.name)
    return rule.name, rule.predict


def newest(spark, table: str, column: str, where: str = "TRUE") -> pd.Timestamp | None:
    """The newest `column` of `table`, naive SGT (read as epoch seconds: freshness.sgt says why)."""
    row = spark.sql(
        f"SELECT unix_timestamp(max({column})) FROM {table} WHERE {where}"
    ).collect()
    return sgt(row[0][0]) if row else None


def feature_rows(
    spark, gold: str, silver: str, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    """Every area at every tick in [first, last], with its gold.area_features columns (null where gold would have no
    row), computed from silver as flood-gold computes them. Lightning, which the models don't read yet
    (contract.HELD_BACK_FEATURES), is left null."""
    from pyspark.sql import functions as F

    lo, hi = _epoch(first), _epoch(last)
    readings = spark.read.table(f"{silver}.rainfall_5min").where(
        FT.seconds("ts").between(lo - FT.RAIN_REACH_S, hi - FT.LAG_S)
    )
    forecasts = spark.read.table(f"{silver}.forecasts_2h").where(
        FT.seconds("updated_at").between(lo - FT.FORECAST_REACH_S, hi)
    )
    weather = FT.tick_weather(
        spark.read.table(f"{silver}.weather_1min").where(
            FT.seconds("ts").between(lo - FT.WEATHER_REACH_S, hi - FT.LAG_S)
        )
    )
    tide = spark.read.table(f"{gold}.tide_levels").where(
        FT.seconds("ts").between(lo, hi)
    )
    reach = spark.read.table(f"{gold}.area_rain_stations")
    areas = spark.read.table(f"{silver}.forecast_areas")
    rain = FT.gauge_rain(readings)
    nearby = FT.nearby_rain(
        rain, areas, spark.read.table(f"{silver}.rain_stations"), weather
    )
    rows = FT.area_features(
        rain, reach, FT.forecast_ticks(forecasts), tide, nearby, weather
    ).where(FT.seconds("ts").between(lo, hi))
    grid = areas.select("area").crossJoin(
        spark.range(1)
        .select(FT.ticks_between(F.lit(lo), F.lit(hi)).alias("tick_s"))
        .select(F.timestamp_seconds("tick_s").alias("ts"))
    )
    frame = grid.join(rows, ["area", "ts"], "left").toPandas()
    return frame.sort_values(["ts", "area"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from pyspark.sql import SparkSession

    from .train import write_predictions

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set(
        "spark.sql.session.timeZone", "Asia/Singapore"
    )  # naive pandas times are SGT
    gold, silver = (
        f"{args.catalog}.{args.gold_schema}",
        f"{args.catalog}.{args.silver_schema}",
    )
    table = f"{args.catalog}.{args.ml_schema}.{C.PREDICTIONS}"
    full_name = f"{args.catalog}.{args.ml_schema}.{args.model}"

    name, predict = live_model(full_name, args.model, args.experiment)
    now = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None)
    reading = newest(spark, f"{silver}.rainfall_5min", "ts")
    scored = (
        newest(spark, table, "ts", f"model = '{name}'")
        if spark.catalog.tableExists(table)
        else None
    )
    span = ticks_to_score(
        now, reading, scored, pd.Timedelta(minutes=args.catch_up_minutes)
    )
    summary = {
        "model": name,
        "now": str(now),
        "newest_reading": str(reading),
        "last_scored": str(scored),
    }
    if span is None:
        log.info("nothing new to score: %s", json.dumps(summary))
        return 0
    first, last = span
    rows = feature_rows(spark, gold, silver, first, last)
    preds = predict(rows)
    write_predictions(spark, table, preds, ticks=(first, last))
    summary |= {
        "first": str(first),
        "last": str(last),
        "rows": len(preds),
        "areas_with_features": int(rows.groupby("ts")["rain_gauges"].count().min()),
        "warned": int((preds["tier"] != "Low").sum()),
        "lag_min": float((now - last) / pd.Timedelta(minutes=1)),
    }
    log.info("scored: %s", json.dumps(summary))
    return 0


def _epoch(ts: pd.Timestamp) -> int:
    """Epoch seconds of a naive SGT time."""
    return int(ts.tz_localize("Asia/Singapore").timestamp())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ml-schema", default="ml")
    parser.add_argument(
        "--model",
        default=C.REGISTERED_MODEL,
        help="registered model name in <catalog>.<ml schema>",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="flood-train's MLflow experiment, for the rule fallback",
    )
    parser.add_argument("--catch-up-minutes", type=int, default=CATCH_UP_MINUTES)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
