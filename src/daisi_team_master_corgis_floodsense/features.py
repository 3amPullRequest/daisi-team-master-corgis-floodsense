"""The feature definitions of gold.area_forecasts and gold.area_features, as Spark DataFrame functions shared by the
flood-gold pipeline (every tick since 2021) and flood-live's score task (the newest ticks). One definition, so a live
prediction reads exactly what the models were trained on.

What gold.area_features is built from, in the order the functions here are applied:

- `forecast_ticks`: the 2-hour forecast valid at each tick (gold.area_forecasts).
- `gauge_rain`: per gauge and tick, the rain totals over each window, over the hours before the last one
  (`prior_rain`), the rain still to drain at each drain rate (`rain_surplus`), and the drains' flow and overflow
  (`drain_response`) (gold.gauge_rain).
- `tick_weather`: the island-wide weather at each tick, the dew point among it, and how complete the lightning feed was
  (gold.tick_weather).
- `nearby_rain`: the rain 5-20 km from each area, and upwind of it (gold.area_nearby_rain); it reads the wind.
- `nearby_alerts`: how many PUB alerts were just posted for other areas near each area (gold.area_nearby_alerts), and
  `recorded_posts`, the ticks whose posts were all recovered, without which a count is unknown.
- `area_features`: all of them, and the tide, per area and tick, with how long the storm over the area has gone on
  and how far the drains' flows are above their highest in it (`storm_age`), the forecast's rain class in fewer
  groups, and the area's ground (gold.area_terrain, from the national map: the same at every tick, so live scoring
  reads gold's table).

Lightning (gold.area_lightning) stays in the gold pipeline: it is held back from the models (contract.HELD_BACK_FEATURES),
so live scoring leaves its columns null.

Times are worked in epoch seconds, so no result depends on the session time zone. SGT has kept UTC+8 since 1982, so an
SGT midnight is the UTC one minus 8 hours, and the 5-minute tick grid is the same in both.
"""

from __future__ import annotations

from datetime import date, datetime

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .ml import contract as C

TICK_S = C.TICK_MINUTES * 60
LAG_S = C.READING_LAG_MINUTES * 60
SLOTS = max(C.RAIN_WINDOWS_MINUTES) // C.TICK_MINUTES  # readings in the longest window
# how many ticks' windows hold a weather reading
WEATHER_SLOTS = C.WEATHER_MINUTES // C.TICK_MINUTES
LIGHTNING_S = C.LIGHTNING_MINUTES * 60
# how far back a tick's inputs reach: its longest rain window ends READING_LAG before it, its weather window likewise,
# and a 2-hour forecast is valid at most 2 hours after its publication
RAIN_REACH_S = LAG_S + SLOTS * TICK_S
WEATHER_REACH_S = LAG_S + WEATHER_SLOTS * TICK_S
FORECAST_REACH_S = 2 * 3600
# the prior rain sums a gauge's hour totals (its longest window, w60) at whole hours before the tick, back to the
# longest of PRIOR_RAIN_HOURS: that much further than the rain windows
HOUR_S = 3600
HOUR_TOTAL = f"w{max(C.RAIN_WINDOWS_MINUTES)}"
assert max(C.RAIN_WINDOWS_MINUTES) * 60 == HOUR_S, "the prior rain steps back by the longest rain window"
PRIOR_RAIN_REACH_S = (max(C.PRIOR_RAIN_HOURS) - 1) * HOUR_S
NEARBY = f"w{C.NEARBY_RAIN_MINUTES}"  # the gauge window the nearby and upwind rain read
POST_LAG_S = C.POST_LAG_MINUTES * 60
ALERT_WINDOW_S = max(m for _, m in C.NEARBY_ALERT_WINDOWS) * 60
# how far back a tick's nearby alerts reach: posts up to the longest window before its post lag
NEARBY_ALERT_REACH_S = POST_LAG_S + ALERT_WINDOW_S
LIGHTNING = tuple(C.lightning_column(km) for km in C.LIGHTNING_RADII_KM)
# the SGT midnight lightning starts, in epoch seconds
LIGHTNING_START_S = (C.LIGHTNING_START - date(1970, 1, 1)).days * 86400 - 8 * 3600
EXACT = "decimal(10,4)"  # weather means are of exact decimals, so they don't depend on how Spark splits the work


def sgt_seconds(t: datetime) -> int:
    """Epoch seconds of a naive SGT time."""
    return int((t - datetime(1970, 1, 1)).total_seconds()) - 8 * 3600


# NEA's bulk re-publication of 8 May 2025 (contract.FORECASTS_REPUBLISHED_SGT), [start, end) in epoch seconds
REPUBLISHED_S = tuple(sgt_seconds(t) for t in C.FORECASTS_REPUBLISHED_SGT)
REPUBLISHED_DELAY_S = C.REPUBLISHED_FORECAST_DELAY_MINUTES * 60


def seconds(ts: str) -> Column:
    """Epoch seconds of a TIMESTAMP column."""
    return F.unix_timestamp(ts)


def published_s() -> Column:
    """When a live fetch could first have read a silver.forecasts_2h row, in epoch seconds: its updated_at, except for
    the forecasts NEA re-published in bulk on 8 May 2025, which count from REPUBLISHED_DELAY after their issue
    (contract.FORECASTS_REPUBLISHED_SGT says why)."""
    updated, issued = seconds("updated_at"), seconds("issued_at")
    rebuilt = (
        (updated >= REPUBLISHED_S[0])
        & (updated < REPUBLISHED_S[1])
        & (updated - issued > FORECAST_REACH_S)
    )
    return F.when(rebuilt, issued + REPUBLISHED_DELAY_S).otherwise(updated)


def tick_at_or_after(s: Column) -> Column:
    return F.ceil(s / TICK_S) * TICK_S


def tick_after(s: Column) -> Column:
    return F.floor(s / TICK_S) * TICK_S + TICK_S


def tick_before(s: Column) -> Column:
    return F.ceil(s / TICK_S) * TICK_S - TICK_S


def ticks_between(first: Column, last: Column) -> Column:
    """One row per tick from `first` to `last` (epoch seconds on the grid, both included); none if last < first."""
    step = F.lit(TICK_S).cast("bigint")
    return F.explode(
        F.when(
            last >= first, F.sequence(first.cast("bigint"), last.cast("bigint"), step)
        )
    )


def distance_km(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    """Haversine, as locations.haversine_km."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    h = F.pow(F.sin((p2 - p1) / 2), 2) + F.cos(p1) * F.cos(p2) * F.pow(
        F.sin(F.radians(lon2 - lon1) / 2), 2
    )
    return F.lit(2 * 6371.0088) * F.asin(F.sqrt(h))


def bearing_deg(lat1: str, lon1: str, lat2: str, lon2: str) -> Column:
    """Initial bearing from point 1 to point 2 (columns named), degrees clockwise from north."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dl = F.radians(F.col(lon2) - F.col(lon1))
    east = F.sin(dl) * F.cos(p2)
    north = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dl)
    return F.pmod(F.degrees(F.atan2(east, north)), F.lit(360.0))


def wind(u: str = "wind_u_kn", v: str = "wind_v_kn") -> tuple[Column, Column]:
    """(speed in knots, the direction it blows from in degrees) of a wind vector."""
    speed = F.sqrt(F.col(u) * F.col(u) + F.col(v) * F.col(v))
    blowing_from = F.pmod(F.degrees(F.atan2(F.col(u), F.col(v))) + 180, F.lit(360.0))
    return speed, blowing_from


def forecast_ticks(forecasts: DataFrame) -> DataFrame:
    """gold.area_forecasts from silver.forecasts_2h rows: each area's newest forecast published (updated_at, or its
    issue for those NEA re-published in bulk: `published_s`) by each tick, while its valid period lasts, with its rain
    class and age."""
    published = forecasts.where(
        "area IS NOT NULL AND updated_at IS NOT NULL AND valid_to IS NOT NULL AND rain_class IS NOT NULL"
    ).select(
        "area",
        "rain_class",
        published_s().alias("published_s"),
        seconds("issued_at").alias("issued_s"),
        seconds("valid_to").alias("valid_to_s"),
    )
    # the newest wins from its publication on; of two published at once, the later issue
    newer = F.lead("published_s").over(
        Window.partitionBy("area").orderBy("published_s", "issued_s")
    )
    # least() skips the null after the newest publication
    spans = published.withColumn("until_s", F.least(newer, F.col("valid_to_s")))
    ticks = spans.select(
        "area",
        "rain_class",
        "published_s",
        ticks_between(
            tick_at_or_after(F.col("published_s")), tick_before(F.col("until_s"))
        ).alias("tick_s"),
    )
    return ticks.select(
        "area",
        F.timestamp_seconds("tick_s").alias("ts"),
        F.col("rain_class").cast("int").alias("forecast_rain_class"),
        ((F.col("tick_s") - F.col("published_s")) / 60)
        .cast("double")
        .alias("forecast_age_min"),
    )


def gauge_rain(readings: DataFrame) -> DataFrame:
    """Per gauge and tick (station_id, tick_s), its rain totals over each of RAIN_WINDOWS_MINUTES (w15, w30, w60), from
    silver.rainfall_5min rows. A reading counts at the ticks whose windows hold it, from READING_LAG after its timestamp
    and not before NEA published it where NEA says when: each reading is expanded into those ticks as slot 0, 1, 2 ...,
    so a window's total is a plain group-by and cannot contain a reading its live twin hadn't yet received. The rain
    surplus reads the readings that arrive at each tick (`arrived`): those counted there for the first time."""
    readings = readings.where(
        "rain_mm IS NOT NULL AND ts IS NOT NULL AND station_id IS NOT NULL"
    )
    # slot k: the reading is the kth newest in the windows of the tick it lands on
    counted = readings.select(
        "station_id",
        F.col("rain_mm").cast("decimal(9,2)").alias("mm"),
        seconds("published_at").alias("published_s"),
        seconds("ts").alias("reading_s"),
        F.explode(F.sequence(F.lit(0), F.lit(SLOTS - 1))).alias("slot"),
    ).withColumn("tick_s", F.col("reading_s") + LAG_S + F.col("slot") * TICK_S)
    counted = counted.where(
        F.col("published_s").isNull() | (F.col("published_s") <= F.col("tick_s"))
    )
    # the slot a reading first counts at: 0, or a later one where NEA published it after the lag
    late = F.ceil((F.col("published_s") - F.col("reading_s") - LAG_S) / TICK_S)
    first_slot = F.when(F.col("published_s").isNull(), 0).otherwise(F.greatest(F.lit(0), late))
    totals = counted.groupBy("station_id", "tick_s").agg(
        *(
            F.sum(F.when(F.col("slot") < w // C.TICK_MINUTES, F.col("mm"))).alias(
                f"w{w}"
            )
            for w in C.RAIN_WINDOWS_MINUTES
        ),
        F.sum(F.when(F.col("slot") == first_slot, F.col("mm"))).alias("arrived"),
    )
    return drain_response(rain_surplus(prior_rain(totals)))


def prior_rain(totals: DataFrame) -> DataFrame:
    """Per gauge and tick, the rain of the hours before the last one (p3, p6, p24 for PRIOR_RAIN_HOURS) beside its
    window totals: the sum of the gauge's hour totals (w60) at the ticks 1, 2, ... hours before, null unless the gauge
    has a row at every one of them. Each hour total is as NEA had published it by its own tick, so the sum holds nothing
    a live fetch at the tick couldn't have had. The ticks a whole number of hours apart share their second of the hour,
    so one window per gauge and second of the hour, ordered by tick, reaches them with a range frame."""
    by_hour = Window.partitionBy("station_id", F.col("tick_s") % HOUR_S).orderBy("tick_s")
    columns = []
    for hours in C.PRIOR_RAIN_HOURS:
        before = by_hour.rangeBetween(-(hours - 1) * HOUR_S, -HOUR_S)
        complete = F.count(HOUR_TOTAL).over(before) == hours - 1
        columns.append(F.when(complete, F.sum(HOUR_TOTAL).over(before)).alias(f"p{hours}"))
    return totals.select("*", *columns)


def rain_surplus(totals: DataFrame) -> DataFrame:
    """Per gauge and tick, the rain still to drain at each of DRAIN_RATES_MM_H (s10, s25, s50, s100) beside its other
    columns: a leaky bucket that the readings arriving at each tick (`arrived`) fill and that drains the rate between
    ticks, never below empty, S(t) = max(0, S(t - 1) + arrived(t) - drain).

    Unrolled, that recursion is the arrivals so far less the drain so far (Q), less the lowest Q up to the tick: S(t) =
    Q(t) - min Q(k) over k <= t, from an empty bucket before the gauge's first row (Lindley's reflected sum). So two
    running windows per gauge compute it. A tick without a row (the gauge was down) arrives nothing and only drains, so
    the lowest Q in a gap is at its last tick: Q of the row after it, less that row's arrivals, plus a tick's drain. Each
    row brings that value into the minimum besides its own Q. Readings are in hundredths of a millimetre and a tick is a
    twelfth of an hour, so in 1/1200 mm every term is a whole number, and a rebuild gives the same values however Spark
    splits the work. Live scoring's buckets start empty a day back (PRIOR_RAIN_REACH_S), and match gold's wherever the
    bucket emptied within that day."""
    so_far = (
        Window.partitionBy("station_id")
        .orderBy("tick_s")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    units = 1200  # per mm
    tick = F.floor(F.col("tick_s") / TICK_S)
    staged = totals.withColumn(
        "_in", (F.coalesce(F.col("arrived"), F.lit(0)) * units).cast("bigint")
    ).withColumn("_in_so_far", F.sum("_in").over(so_far))
    columns = []
    for rate in C.DRAIN_RATES_MM_H:
        drain = rate * units // 12  # per tick
        q = F.col("_in_so_far") - F.lit(drain) * tick
        lowest = F.min(q - F.greatest(F.col("_in") - drain, F.lit(0))).over(so_far)
        columns.append(((q - lowest) / units).cast("double").alias(f"s{rate}"))
    return staged.select("*", *columns).drop("_in", "_in_so_far")


def drain_response(totals: DataFrame) -> DataFrame:
    """Per gauge and tick, the drains' response beside its other columns (contract.DRAIN_STATE, as d10, d30 for the
    flows and o10, o30 for the overflows): `gauge_drains` over each gauge's ticks, in order. A recursion that isn't
    linear has no closed form in windows, so it runs in pandas, a gauge per group (about 50 MB at most, so unlike the
    tide fit it fits a serverless worker), and joins back."""
    keys = ["station_id", "tick_s"]
    arriving = totals.select(
        *keys, F.coalesce(F.col("arrived"), F.lit(0)).cast("double").alias("mm")
    )
    schema = T.StructType(
        [
            *(totals.schema[k] for k in keys),
            *(T.StructField(c, T.DoubleType()) for c in DRAIN_COLUMNS),
        ]
    )
    drains = arriving.groupBy("station_id").applyInPandas(gauge_drains, schema)
    return totals.join(drains, keys, "left")


def _gauge_drains():
    """`gauge_drains`, which carries its constants with it: the executors can't import this package, and a function
    defined inside another travels to them by value (cloudpickle), with the names it closes over."""
    tick_s, tick_min = TICK_S, C.TICK_MINUTES
    m, step = C.DRAIN_EXPONENT, C.DRAIN_FLOW_STEP_MM_H
    reference = C.DRAIN_REFERENCE_MM_H / 60  # mm a minute
    # outflow a * S^m mm a minute, with time constant k at the reference rain, where S = m * reference * k
    a = [reference / (m * reference * k) ** m for k in C.DRAIN_RESPONSE_MINUTES]
    grows = [(m - 1) * x * tick_min for x in a]  # a tick without rain adds this to S^(1 - m): the exact decay
    up, down = 1 - m, 1 / (1 - m)
    # what a tick's outflow is in steps of flow (mm/h), and the step of flow and a tick of its overflow in mm, so the
    # columns come out as exact as a double allows
    per_tick = 60 / (tick_min * step)
    capacity = round(C.DRAIN_CAPACITY_MM_H / step)
    per_mm_h, per_mm = round(1 / step), round(per_tick)
    loss, dries = C.DRAIN_INITIAL_LOSS_MM, C.DRAIN_LOSS_RECOVERY_MM_DAY * tick_min / (24 * 60)
    names = DRAIN_COLUMNS

    def gauge_drains(rows):
        """One gauge's rows (station_id, tick_s and mm, the rain arriving at the tick), in any order: each tick's
        drains (d10, d30, o10, o30), by the recursion of contract.py. A tick without a row (the gauge was down) has no
        rain. At each tick the hollows dry a little, and the tick's rain fills them first and the stores with the rest.
        A store's flow is what it lets out over the next tick from what it holds, as dS/dt = -a S^m does exactly, in
        mm/h rounded to a step, so under steady rain it lets out what comes in. A store whose flow rounds to 0 without
        new rain has drained. The flow beyond capacity fills the overflow, in whole steps."""
        import math

        import numpy as np
        import pandas as pd

        rows = rows.sort_values("tick_s")
        ticks = rows["tick_s"].to_numpy(dtype="int64") // tick_s
        arriving = rows["mm"].to_numpy(dtype="float64")
        n, kinds = len(rows), len(a)
        out = np.zeros((2 * kinds, n))
        # each store as the tick's rain leaves it, and as it will be at the next tick without more
        stored, then = [0.0] * kinds, [0.0] * kinds
        flow, waiting = [0] * kinds, [0] * kinds
        wet = 0.0  # held in the ground's hollows

        def advance(rain, new_tick):
            nonlocal wet
            if new_tick:
                wet = max(0.0, wet - dries)
            held = min(rain, loss - wet)
            wet += held
            rain -= held
            for j in range(kinds):
                s = (then[j] if new_tick else stored[j]) + rain
                after = (s**up + grows[j]) ** down if s > 0.0 else 0.0
                flow[j] = math.floor((s - after) * per_tick + 0.5)
                if flow[j] == 0 and rain == 0.0:
                    s = after = 0.0
                stored[j], then[j] = s, after
                if new_tick:
                    waiting[j] = max(0, waiting[j] + flow[j] - capacity)

        last = None
        for i in range(n):
            gap = 1 if last is None else int(ticks[i] - last)
            last = ticks[i]
            for _ in range(gap - 1):  # the ticks the gauge missed: no rain
                if wet == 0.0 and not any(stored) and not any(waiting):
                    break  # all empty, and they stay so
                advance(0.0, True)
            # two rows in one tick (off the grid) arrive together
            advance(arriving[i], gap > 0)
            for j in range(kinds):
                out[j, i] = flow[j] / per_mm_h
                out[kinds + j, i] = waiting[j] / per_mm
        frame = pd.DataFrame(out.T, columns=list(names))
        frame.insert(0, "tick_s", rows["tick_s"].to_numpy(dtype="int64"))
        frame.insert(0, "station_id", rows["station_id"].to_numpy(dtype=object))
        return frame.astype({"station_id": object})  # not pandas 3's Arrow strings, which applyInPandas can't send back

    return gauge_drains


DRAIN_COLUMNS = (
    *(f"d{k}" for k in C.DRAIN_RESPONSE_MINUTES),
    *(f"o{k}" for k in C.DRAIN_RESPONSE_MINUTES),
)
gauge_drains = _gauge_drains()


def storm_age(rows: DataFrame) -> DataFrame:
    """Beside per-area rows (area, tick_s, the rain windows and the drains' flows), how long the storm over the area
    has gone on, in minutes (contract.STORM_AGE): from the tick that started it, the first whose STORM_WINDOW_MINUTES
    total was above 0 after STORM_GAP_MINUTES without one, to the tick, while the last wet tick is at most
    STORM_GAP_MINUTES back, and at most STORM_MAX_MINUTES; else null. A tick without a row (no gauge in reach reported)
    is dry. With it, how far each flow is above its highest at the storm's earlier ticks (contract.DRAIN_RISE), as far
    back as STORM_MAX_MINUTES, so live scoring's day of readings sees all of them; 0 before the storm's first; null
    outside a storm. Each tick's rain is point in time, so both read only what the area's earlier ticks had seen."""
    gap = C.STORM_GAP_MINUTES * 60
    by_area = Window.partitionBy("area").orderBy("tick_s")
    wet = F.when(F.col(C.rain_column(C.STORM_WINDOW_MINUTES)) > 0, F.col("tick_s"))
    before = F.last(wet, ignorenulls=True).over(
        by_area.rowsBetween(Window.unboundedPreceding, -1)
    )
    starts = rows.select(
        "*",
        wet.alias("_wet_s"),
        F.when(
            wet.isNotNull() & (before.isNull() | (F.col("tick_s") - before > gap)),
            F.col("tick_s"),
        ).alias("_start_s"),
    )
    so_far = by_area.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    storms = starts.select(
        "*",
        F.last("_start_s", ignorenulls=True).over(so_far).alias("_storm_s"),
        F.last("_wet_s", ignorenulls=True).over(so_far).alias("_last_wet_s"),
    )
    in_storm = F.col("tick_s") - F.col("_last_wet_s") <= gap
    age = F.when(
        in_storm,
        F.least(
            (F.col("tick_s") - F.col("_storm_s")) / 60,
            F.lit(float(C.STORM_MAX_MINUTES)),
        ),
    )
    earlier = (
        Window.partitionBy("area", "_storm_s")
        .orderBy("tick_s")
        .rangeBetween(-C.STORM_MAX_MINUTES * 60, -1)
    )
    rises = [
        F.when(
            in_storm,
            F.round(F.col(flow) - F.coalesce(F.max(flow).over(earlier), F.lit(0.0)), 1),
        ).alias(rise)
        for flow, rise in zip(C.DRAIN_FLOW, C.DRAIN_RISE)
    ]
    return storms.select("*", age.cast("double").alias(C.STORM_AGE), *rises).drop(
        "_wet_s", "_start_s", "_storm_s", "_last_wet_s"
    )


def tick_weather(
    readings: DataFrame, lightning_records: DataFrame | None = None
) -> DataFrame:
    """gold.tick_weather from silver.weather_1min rows and silver.lightning_records rows: per tick, the mean air
    temperature, humidity, dew point and wind (u eastward, v northward, knots) over NEA's stations, each station's
    newest reading in the WEATHER_MINUTES ending READING_LAG before the tick that NEA had published by it; the stations
    behind each mean; and how many of the lightning records of the last LIGHTNING_MINUTES exist (null without
    `lightning_records`).

    Wind speed and direction are combined into u and v per station before averaging, because directions can't be
    averaged as numbers. NEA gives the direction the wind blows from, so the vector points the other way. The dew point
    is computed per station too, as it isn't linear in the temperature and humidity it comes from."""
    readings = readings.where(
        "value IS NOT NULL AND ts IS NOT NULL AND station_id IS NOT NULL"
    )
    # slot k: the reading is in the window of the kth tick from the first whose window holds it
    counted = readings.select(
        "station_id",
        "variable",
        "value",
        seconds("ts").alias("reading_s"),
        seconds("published_at").alias("published_s"),
        F.explode(F.sequence(F.lit(0), F.lit(WEATHER_SLOTS - 1))).alias("slot"),
    ).withColumn(
        "tick_s",
        tick_at_or_after(F.col("reading_s")) + LAG_S + F.col("slot") * TICK_S,
    )
    counted = counted.where(
        F.col("published_s").isNull() | (F.col("published_s") <= F.col("tick_s"))
    )
    newest = counted.groupBy("station_id", "variable", "tick_s").agg(
        F.max_by("value", "reading_s").alias("value")
    )
    per_station = (
        newest.groupBy("station_id", "tick_s")
        .pivot("variable", list(C.WEATHER_READINGS))
        .agg(F.first("value"))
    )
    speed, blowing_from = F.col("wind_speed_kn"), F.radians("wind_dir_deg")
    windy = speed.isNotNull() & blowing_from.isNotNull()
    per_tick = per_station.groupBy("tick_s").agg(
        F.avg(F.col("air_temp_c").cast(EXACT)).cast("double").alias("air_temp_c"),
        F.avg(F.col("rel_humidity_pct").cast(EXACT))
        .cast("double")
        .alias("rel_humidity_pct"),
        F.avg(dew_point("air_temp_c", "rel_humidity_pct").cast(EXACT))
        .cast("double")
        .alias("dew_point_c"),
        F.avg(F.when(windy, -speed * F.sin(blowing_from)).cast(EXACT))
        .cast("double")
        .alias("wind_u_kn"),
        F.avg(F.when(windy, -speed * F.cos(blowing_from)).cast(EXACT))
        .cast("double")
        .alias("wind_v_kn"),
        F.count("air_temp_c").alias("temp_stations"),
        F.count("rel_humidity_pct").alias("humidity_stations"),
        F.count(F.when(windy, 1)).alias("wind_stations"),
    )
    if lightning_records is None:
        rows, records = per_tick, F.lit(None)
    else:
        # a record counts for the ticks whose window (t - LIGHTNING_MINUTES, t] holds it
        record_s = seconds("record_ts")
        counts = (
            lightning_records.where("record_ts IS NOT NULL")
            .select(
                ticks_between(
                    tick_at_or_after(record_s), tick_before(record_s + LIGHTNING_S)
                ).alias("tick_s")
            )
            .groupBy("tick_s")
            .agg(F.count("*").alias("lightning_records"))
        )
        # a full join, so a tick with only lightning records still has a row
        rows = per_tick.join(counts, "tick_s", "full")
        records = F.coalesce("lightning_records", F.lit(0))
    return rows.select(
        F.timestamp_seconds("tick_s").alias("ts"),
        "air_temp_c",
        "rel_humidity_pct",
        "dew_point_c",
        "wind_u_kn",
        "wind_v_kn",
        *(
            F.coalesce(c, F.lit(0)).cast("int").alias(c)
            for c in ("temp_stations", "humidity_stations", "wind_stations")
        ),
        records.cast("int").alias("lightning_records"),
    )


def dew_point(temp: str, humidity: str) -> Column:
    """The Magnus dew point (deg C) of a station's air temperature and relative humidity (contract.dew_point_c); null
    unless both are known and the humidity is above 0."""
    t, rh = F.col(temp).cast("double"), F.col(humidity).cast("double")
    gamma = F.log(rh / 100) + C.MAGNUS_A * t / (C.MAGNUS_B_C + t)
    return F.when(t.isNotNull() & (rh > 0), C.MAGNUS_B_C * gamma / (C.MAGNUS_A - gamma))


def nearby_rain(
    rain: DataFrame, areas: DataFrame, gauges: DataFrame, weather: DataFrame
) -> DataFrame:
    """gold.area_nearby_rain: per area and tick (area, tick_s), the largest NEARBY_RAIN_MINUTES total among the gauges
    NEARBY_RAIN_KM away (`nearby`), and among those of them upwind (`upwind`). From gauge_rain's rows,
    silver.forecast_areas, silver.rain_stations and tick_weather's rows. A gauge is upwind when its bearing from the
    area is within UPWIND_HALF_ANGLE_DEG of the direction the island-wide wind blows from, and the wind is at least
    UPWIND_MIN_WIND_KN. Rows only where one of the gauges rained; area_features reads a missing row as none."""
    areas = areas.select(
        "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
    )
    near, far = C.NEARBY_RAIN_KM
    km = distance_km(F.col("area_lat"), F.col("area_lon"), F.col("lat"), F.col("lon"))
    gauges = gauges.select("station_id", "lat", "lon")
    pairs = (
        areas.crossJoin(gauges)  # 47 areas x ~100 gauges
        .select(
            "area",
            "station_id",
            km.alias("km"),
            bearing_deg("area_lat", "area_lon", "lat", "lon").alias("bearing"),
        )
        .where((F.col("km") > near) & (F.col("km") <= far))
    )
    wet = rain.where(F.col(NEARBY) > 0).select("station_id", "tick_s", NEARBY)
    speed, blowing_from = wind()
    winds = weather.where("wind_u_kn IS NOT NULL").select(
        seconds("ts").alias("tick_s"),
        speed.alias("wind_kn"),
        blowing_from.alias("blowing_from"),
    )
    rows = wet.join(F.broadcast(pairs), "station_id").join(winds, "tick_s", "left")
    off_wind = F.abs(F.pmod(F.col("bearing") - F.col("blowing_from") + 180, 360) - 180)
    upwind = (F.col("wind_kn") >= C.UPWIND_MIN_WIND_KN) & (
        off_wind <= C.UPWIND_HALF_ANGLE_DEG
    )
    return rows.groupBy("area", "tick_s").agg(
        F.max(NEARBY).alias("nearby"),
        F.max(F.when(upwind, F.col(NEARBY))).alias("upwind"),
    )


def nearby_alerts(alerts: DataFrame, areas: DataFrame) -> DataFrame:
    """gold.area_nearby_alerts: per area and tick (area, tick_s), how many PUB alerts were posted for other areas at
    places near it, in each of contract.NEARBY_ALERT_WINDOWS (contract.PUB_ALERTS). From silver.area_alerts (an alert
    without a posting time counts from its stamp) and silver.forecast_areas. An alert counts at the ticks t with
    t - POST_LAG - window < posted <= t - POST_LAG. Rows only where one counted; area_features reads a missing row as
    none."""
    far = max(km for km, _ in C.NEARBY_ALERT_WINDOWS)
    areas = areas.select(
        "area", F.col("lat").alias("area_lat"), F.col("lon").alias("area_lon")
    )
    posted = alerts.where("lat IS NOT NULL AND lon IS NOT NULL").select(
        F.col("area").alias("alert_area"),
        F.unix_timestamp(F.coalesce("posted_ts", "alert_ts")).alias("posted_s"),
        "lat",
        "lon",
    )
    km = distance_km(F.col("area_lat"), F.col("area_lon"), F.col("lat"), F.col("lon"))
    pairs = (
        posted.crossJoin(F.broadcast(areas))  # a few thousand alerts x 47 areas
        .where(F.col("area") != F.col("alert_area"))
        .select("area", "posted_s", km.alias("km"))
        .where(F.col("km") <= far)
    )
    counts_from = F.col("posted_s") + POST_LAG_S
    first = tick_at_or_after(counts_from)
    last = F.floor((counts_from + ALERT_WINDOW_S - 1) / TICK_S) * TICK_S
    rows = pairs.select(
        "area",
        "posted_s",
        "km",
        F.explode(F.sequence(first, last, F.lit(TICK_S).cast("long"))).alias("tick_s"),
    )
    return rows.groupBy("area", "tick_s").agg(
        *(
            F.sum(
                F.when(
                    (F.col("km") <= km) & (F.col("tick_s") < counts_from + minutes * 60), 1
                ).otherwise(0)
            )
            .cast("int")
            .alias(C.nearby_alerts_column(km, minutes))
            for km, minutes in C.NEARBY_ALERT_WINDOWS
        )
    )


def recorded_posts(label_ticks: DataFrame) -> DataFrame:
    """The ticks (tick_s) whose nearby-alert windows lost no post, from gold.label_ticks: those ALERT_WINDOW_S after a
    tick whose label window was recorded. A label window holds every post from STAMP_AHEAD_MINUTES before its tick to
    HORIZON + POST_DELAY_MAX after it, so the tick an hour later has every post its windows read (contract.py)."""
    return label_ticks.where(F.col("label_window") == C.RECORDED).select(
        (seconds("ts") + ALERT_WINDOW_S).alias("tick_s")
    )


def area_features(
    rain: DataFrame,
    reach: DataFrame,
    forecasts: DataFrame,
    tide: DataFrame,
    nearby: DataFrame,
    weather: DataFrame,
    lightning: DataFrame | None = None,
    *,
    alerts_nearby: DataFrame | None = None,
    posts_recorded: DataFrame | None = None,
    terrain: DataFrame | None = None,
) -> DataFrame:
    """gold.area_features (contract.AREA_FEATURES_COLUMNS, in its order) from gauge_rain's rows,
    gold.area_rain_stations, forecast_ticks' rows, gold.tide_levels, nearby_rain's rows, tick_weather's rows and
    gold.area_lightning (or none: its columns are then null). Per area and tick, the largest total among the gauges in
    reach for each window (the max, not the mean, because a flash flood follows the most intense cell), joined to the
    rest. A row exists only where a gauge in reach had a reading in the last hour. The optional columns are null where
    their source had nothing: no lightning before its feed began or when it was down for part of the window, no upwind
    rain without a wind, no storm age outside a storm. PUB's alerts nearby come from nearby_alerts' rows (none: null);
    with `posts_recorded` (recorded_posts' rows, as gold passes) they are null at the ticks whose posts may be lost, and
    without it (live scoring, which reads the channel as it posts) they are known at every tick. The ground comes from
    gold.area_terrain (`terrain`, one row per area; none: null), the same at every tick of an area."""
    keys = ["area", "tick_s"]
    per_area = storm_age(
        rain.join(
            F.broadcast(
                reach.where("station_id IS NOT NULL").select("area", "station_id")
            ),
            "station_id",
        )
        .groupBy(*keys)
        .agg(
            *(
                F.max(f"w{w}").cast("double").alias(C.rain_column(w))
                for w in C.RAIN_WINDOWS_MINUTES
            ),
            # the largest among the gauges whose prior hours are all there (max skips the others' nulls)
            *(
                F.max(f"p{h}").cast("double").alias(C.prior_rain_column(h))
                for h in C.PRIOR_RAIN_HOURS
            ),
            *(
                F.max(f"s{r}").alias(C.surplus_column(r))
                for r in C.DRAIN_RATES_MM_H
            ),
            *(
                F.max(f"d{k}").alias(C.drain_flow_column(k))
                for k in C.DRAIN_RESPONSE_MINUTES
            ),
            *(
                F.max(f"o{k}").alias(C.drain_overflow_column(k))
                for k in C.DRAIN_RESPONSE_MINUTES
            ),
            F.count("*").cast("int").alias("rain_gauges"),
        )
    )
    # the forecast's rain class in fewer groups: a map from class to group
    groups = {
        name: F.try_element_at(
            F.create_map(*(F.lit(x) for k, g in enumerate(grouping) for x in (k, g))),
            F.col("forecast_rain_class"),
        )
        .cast("int")
        .alias(name)
        for name, grouping in C.FORECAST_GROUPINGS.items()
    }
    forecast = forecasts.select(
        "area",
        seconds("ts").alias("tick_s"),
        "forecast_rain_class",
        "forecast_age_min",
    )
    weather = weather.select(
        seconds("ts").alias("tick_s"),
        "air_temp_c",
        "rel_humidity_pct",
        "dew_point_c",
        "wind_u_kn",
        "wind_v_kn",
        "lightning_records",
    )
    # the same tide for every area: the model's, not the gauge's (contract.py)
    tides = tide.select(
        seconds("ts").alias("tick_s"),
        F.col("predicted_m").alias("tide_height_m"),
        "minutes_to_high_tide",
    )
    rows = (
        per_area.join(forecast, keys, "left")
        .join(nearby, keys, "left")
        .join(weather, "tick_s", "left")
        .join(tides, "tick_s", "left")
    )
    if lightning is None:
        strikes = [F.lit(None).cast("int").alias(c) for c in LIGHTNING]
    else:
        rows = rows.join(
            lightning.select("area", seconds("ts").alias("tick_s"), *LIGHTNING),
            keys,
            "left",
        )
        # lightning needs the feed to have been up for the whole window
        lightning_up = (F.col("tick_s") >= LIGHTNING_START_S) & (
            F.col("lightning_records") == C.LIGHTNING_RECORDS_PER_WINDOW
        )
        strikes = [
            F.when(lightning_up, F.coalesce(F.col(c), F.lit(0))).cast("int").alias(c)
            for c in LIGHTNING
        ]
    if alerts_nearby is None:
        posted = [F.lit(None).cast("int").alias(c) for c in C.PUB_ALERTS]
    else:
        rows = rows.join(alerts_nearby.select(*keys, *C.PUB_ALERTS), keys, "left")
        known = F.lit(True)
        if posts_recorded is not None:
            rows = rows.join(
                posts_recorded.select("tick_s", F.lit(True).alias("posts_recorded")),
                "tick_s",
                "left",
            )
            known = F.col("posts_recorded").isNotNull()
        posted = [
            F.when(known, F.coalesce(F.col(c), F.lit(0))).cast("int").alias(c)
            for c in C.PUB_ALERTS
        ]
    if terrain is None:
        ground = [F.lit(None).cast("double").alias(c) for c in C.MAP_FEATURES]
    else:
        rows = rows.join(
            F.broadcast(terrain.select("area", *C.MAP_FEATURES)), "area", "left"
        )
        ground = [F.col(c).cast("double").alias(c) for c in C.MAP_FEATURES]
    # upwind needs a wind to look into
    speed, _ = wind()
    return rows.select(
        "area",
        F.timestamp_seconds("tick_s").alias("ts"),
        *(C.rain_column(w) for w in C.RAIN_WINDOWS_MINUTES),
        *C.PRIOR_RAIN,
        *C.STORM_HISTORY,
        *C.DRAIN_RESPONSE,
        "rain_gauges",
        "forecast_rain_class",
        *groups.values(),
        "forecast_age_min",
        F.coalesce(F.col("nearby"), F.lit(0))
        .cast("double")
        .alias("rain_nearby_30min_mm"),
        F.when(speed >= C.UPWIND_MIN_WIND_KN, F.coalesce(F.col("upwind"), F.lit(0)))
        .cast("double")
        .alias("rain_upwind_30min_mm"),
        *posted,
        "air_temp_c",
        "rel_humidity_pct",
        "dew_point_c",
        "wind_u_kn",
        "wind_v_kn",
        *strikes,
        "tide_height_m",
        "minutes_to_high_tide",
        *ground,
    )
