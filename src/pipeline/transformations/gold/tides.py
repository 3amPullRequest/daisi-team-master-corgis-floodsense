"""Gold tide tables: the harmonic model fitted on Tanjong Pagar's gauge (gold.tide_model), its datum checked against
NEA's high and low waters (gold.tide_datum), and a tide for every 5-minute tick (gold.tide_levels), which
gold.area_features reads.

UHSLC's gauge ends in December 2023 and has gaps before that, so gold.tide_levels imputes: the gauge where it read, the
prediction plus the gauge's surge interpolated between two hourly readings, and the prediction alone in the gaps and
from 2024 on. `source` says which. The models read `predicted_m` only (contract.py says why).

The fit and the predictions run in pandas (tides.py, with utide) through applyInPandas: the fit on one task, the
predictions a month per task. Times cross into pandas as epoch seconds.
"""

from datetime import date
from itertools import pairwise

from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F
from utilities.gold import (
    TICK_S,
    seconds,
    sgt_midnight_s,
    silver,
    tick_after,
    ticks_between,
)

from daisi_team_master_corgis_floodsense import tides as T
from daisi_team_master_corgis_floodsense.ml import contract as C

HOUR_S = 3600
FIT_START_S = (
    sgt_midnight_s(T.FIT_START) + 8 * HOUR_S
)  # UTC midnight: UHSLC's hours are GMT
# From the first rain readings to the end of next year, so live scoring always finds its tick.
START = C.TIDE_START
END = date(date.today().year + 2, 1, 1)
SPAN_COLUMNS = "start_s long, end_s long, tick_s long"


def _month_spans() -> list[tuple[int, int, int]]:
    """One span per SGT month from START to END, each in whole ticks (an SGT midnight is on the tick grid)."""
    firsts, day = [], START
    while day < END:
        firsts.append(day)
        day = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    bounds = [sgt_midnight_s(d) for d in (*firsts, END)]
    return [(a, b, TICK_S) for a, b in pairwise(bounds)]


@dp.materialized_view(
    name="tide_model",
    comment="The harmonic tide model for Tanjong Pagar: utide's least-squares fit (default constituents, linear trend, "
    f"nodal corrections) on UHSLC's valid hourly readings from {T.FIT_START} (silver.tide_gauge_hourly), heights in "
    "metres above UHSLC's zero. One row: the fit's span, mean, trend, in-sample residual (the surge the model can't "
    "know), the constituents largest first, and coef_json, which tides.from_json turns back into the fit.",
)
@dp.expect_all(
    {
        "fit_has_years": "observations > 100000",
        "residual_small": "residual_std_m < 0.15",  # 0.089 m on 2026-10-20, against the tide's 0.71 m
    }
)
def tide_model():
    gauge = (
        spark.read.table(silver(spark, "tide_gauge_hourly"))
        .where(F.col("sea_level_m").isNotNull() & (seconds("ts") >= FIT_START_S))
        .select(seconds("ts").alias("ts_s"), "sea_level_m", F.lit(0).alias("fit"))
    )
    fitted = gauge.groupBy("fit").applyInPandas(T.fit_frame, T.FIT_COLUMNS)
    return fitted.select(
        F.timestamp_seconds("fit_start_s").alias("fit_start"),
        F.timestamp_seconds("fit_end_s").alias("fit_end"),
        "observations",
        "mean_m",
        "trend_m_per_year",
        "residual_std_m",
        F.from_json(
            "constituents_json",
            "array<struct<name: string, amplitude_m: double, phase_deg: double>>",
        ).alias("constituents"),
        "coef_json",
    )


@dp.materialized_view(
    name="tide_predictions",
    private=True,
    comment="The model's height (m above UHSLC's zero) at every tick, and its high and low waters to the minute.",
    cluster_by=["ts_s"],
)
def tide_predictions():
    spans = spark.createDataFrame(_month_spans(), SPAN_COLUMNS)
    model = spark.read.table("tide_model").select("coef_json")
    return (
        spans.crossJoin(model)
        .groupBy("start_s")
        .applyInPandas(T.predict_frame, T.PREDICTION_COLUMNS)
    )


@dp.materialized_view(
    name="tide_datum",
    comment="Each NEA high or low water (silver.tide_extremes) beside the model's nearest one of the same kind within "
    "3 hours. datum_offset_m, the median of NEA's height minus the model's, turns the model's heights (UHSLC's zero) "
    "into NEA's (Chart Datum); predicted_m already has it added. NEA prints heights to 0.1 m, so errors up to 0.05 m "
    "are rounding.",
    cluster_by=["ts"],
)
@dp.expect_all(
    {
        "nea_matched": "predicted_ts IS NOT NULL",
        "height_within_tolerance": f"abs(height_error_m) <= {T.HEIGHT_TOLERANCE_M}",
        "time_within_tolerance": f"abs(time_error_min) <= {T.TIME_TOLERANCE_MIN}",
    }
)
def tide_datum():
    nea = spark.read.table(silver(spark, "tide_extremes")).select(
        "ts",
        "kind",
        F.col("height_m").alias("nea_height_m"),
        seconds("ts").alias("nea_s"),
    )
    turns = (
        spark.read.table("tide_predictions")
        .where(F.col("point") != T.TICK)
        .select(
            F.col("point").alias("kind"),
            F.col("ts_s").alias("model_s"),
            F.col("height_m").alias("model_m"),
        )
    )
    pairs = nea.join(
        turns,
        (nea["kind"] == turns["kind"])
        & (F.abs(turns["model_s"] - nea["nea_s"]) <= 3 * HOUR_S),
        "left",
    ).drop(turns["kind"])
    nearest = Window.partitionBy("ts", "kind").orderBy(
        F.abs(F.col("model_s") - F.col("nea_s"))
    )
    matched = pairs.withColumn("rank", F.row_number().over(nearest)).where("rank = 1")
    offset = F.median(F.col("nea_height_m") - F.col("model_m")).over(
        Window.partitionBy()
    )
    with_offset = matched.withColumn("datum_offset_m", offset)
    predicted = F.col("model_m") + F.col("datum_offset_m")
    return with_offset.select(
        "ts",
        "kind",
        "nea_height_m",
        F.timestamp_seconds("model_s").alias("predicted_ts"),
        predicted.alias("predicted_m"),
        (predicted - F.col("nea_height_m")).alias("height_error_m"),
        ((F.col("model_s") - F.col("nea_s")) / 60)
        .cast("double")
        .alias("time_error_min"),
        "datum_offset_m",
    )


@dp.materialized_view(
    name="tide_levels",
    comment=f"The tide at Tanjong Pagar at every 5-minute tick from {START} to {END}, metres above Chart Datum. "
    "tide_m is imputed where the gauge is silent: source 'observed' (a valid UHSLC hourly reading at the tick), "
    "'interpolated' (between two valid hourly readings: the prediction plus the gauge's surge, interpolated), or "
    "'predicted' (gaps, and every tick from 2024 on, when UHSLC's record stops). predicted_m is the model alone, "
    "which the models read; surge_m is tide_m minus it. Heights carry gold.tide_datum's offset (0 if NEA's table "
    "hasn't landed). next_high_ts and minutes_to_high_tide come from the predicted high waters.",
    cluster_by=["ts"],
)
@dp.expect_all_or_fail({"tide_known": "tide_m IS NOT NULL AND predicted_m IS NOT NULL"})
@dp.expect("tide_plausible", "tide_m BETWEEN -0.5 AND 4.5")
def tide_levels():
    predictions = spark.read.table("tide_predictions")
    ticks = predictions.where(F.col("point") == T.TICK).select(
        F.col("ts_s").alias("tick_s"), F.col("height_m").alias("model_m")
    )
    datum = spark.read.table("tide_datum").agg(
        F.coalesce(F.max("datum_offset_m"), F.lit(0.0)).alias("offset_m")
    )
    gauge = spark.read.table(silver(spark, "tide_gauge_hourly")).where(
        "sea_level_m IS NOT NULL"
    )
    # the surge at each valid hour: the gauge minus the model
    surge = gauge.join(ticks, seconds("ts") == ticks["tick_s"]).select(
        F.col("tick_s").alias("hour_s"),
        (F.col("sea_level_m") - F.col("model_m")).alias("surge"),
    )
    before = surge.select(F.col("hour_s").alias("h0_s"), F.col("surge").alias("r0"))
    after = surge.select(
        (F.col("hour_s") - HOUR_S).alias("h0_s"), F.col("surge").alias("r1")
    )
    rows = (
        ticks.withColumn("h0_s", F.floor(F.col("tick_s") / HOUR_S) * HOUR_S)
        .join(before, "h0_s", "left")
        .join(after, "h0_s", "left")
        .crossJoin(datum)
    )
    on_hour = F.col("tick_s") == F.col("h0_s")
    source = (
        F.when(on_hour & F.col("r0").isNotNull(), "observed")
        .when(F.col("r0").isNotNull() & F.col("r1").isNotNull(), "interpolated")
        .otherwise("predicted")
    )
    fraction = (F.col("tick_s") - F.col("h0_s")) / HOUR_S
    surge_at_tick = F.when(source == "observed", F.col("r0")).when(
        source == "interpolated", F.col("r0") + (F.col("r1") - F.col("r0")) * fraction
    )
    predicted = F.col("model_m") + F.col("offset_m")
    levels = rows.select(
        "tick_s",
        (predicted + F.coalesce(surge_at_tick, F.lit(0.0))).alias("tide_m"),
        source.alias("source"),
        predicted.alias("predicted_m"),
        surge_at_tick.alias("surge_m"),
        "offset_m",
    )
    # each tick after a high water, up to and including the next, waits for that next one
    highs = predictions.where(F.col("point") == T.HIGH).select(
        F.col("ts_s").alias("high_s"), F.col("height_m").alias("high_model_m")
    )
    previous = F.lag("high_s").over(Window.orderBy("high_s"))
    waits = highs.withColumn("previous_s", previous).select(
        "high_s",
        "high_model_m",
        ticks_between(
            tick_after(F.col("previous_s")), F.floor(F.col("high_s") / TICK_S) * TICK_S
        ).alias("tick_s"),
    )
    return levels.join(waits, "tick_s", "left").select(
        F.timestamp_seconds("tick_s").alias("ts"),
        "tide_m",
        "source",
        "predicted_m",
        "surge_m",
        F.timestamp_seconds("high_s").alias("next_high_ts"),
        (F.col("high_model_m") + F.col("offset_m")).alias("next_high_m"),
        ((F.col("high_s") - F.col("tick_s")) / 60)
        .cast("double")
        .alias("minutes_to_high_tide"),
    )
