"""What every FloodSense model reads and writes. Build silver and gold backwards from here.

The unit of prediction is one NEA forecast area at one 5-minute tick: how likely is a PUB alert in that area within the
next hour? The rainfall-rule baseline and the learned models (the ordinal model, and the gated LightGBM pair before it)
read the same tables and are judged by the same scoreboard (scoreboard.py). A learned model only reads more feature
columns: it adds columns to `area_features`, never a new grain, label or scoring rule.

Timestamps are SGT; naive ones are read as SGT.

The tables, in the order data flows through them, with the constants that define each below:

- silver.area_alerts (AREA_ALERTS_COLUMNS): PUB's alerts, each placed in a forecast area, once. The only source of
  labels.
- gold.label_ticks and gold.area_labels (LABEL_TICKS_COLUMNS, AREA_LABELS_COLUMNS): for every area and 5-minute tick,
  whether a PUB alert came within the next hour, with the ticks whose label window was never recorded kept apart.
- gold.area_features and gold.feature_ticks (AREA_FEATURES_COLUMNS, FEATURE_TICKS_COLUMNS): what each area had seen by
  each tick, and whether the inputs behind a tick were recorded.
- gold.model_rows (MODEL_ROWS_COLUMNS): the features joined to the labels wherever both windows are recorded. The only
  table the models learn from or are tested on.
- ml.predictions (PREDICTIONS_COLUMNS): each model's tier, probability and reason per area and tick, for the app, the
  dashboard and the scoreboard.

Three rules run through all of it:

- Point in time. A value at tick t is only what a live fetch at t could have had: a reading counts from
  READING_LAG_MINUTES after its timestamp and not before NEA published it, a forecast from its publication. So training
  rows and replays of past storms match what the live model sees.
- Missing is not zero. PUB deleted channel history and NEA's feeds have outages, so some ticks have no recorded
  label or no recorded feature. Those are marked missing, never read as quiet, and kept out of the models and the
  scoreboard (RECORDED/MISSING and the reasons).
- One grain, one split. Every table is per area and tick, the horizon is HORIZON_MINUTES, and the train/embargo/test
  dates are fixed constants computed once from the recorded label time (TRAIN_START, TEST_START, SPLIT_END).
"""

from __future__ import annotations

from datetime import date, datetime

N_AREAS = 47  # NEA 2-hour forecast areas (`area_metadata` of the two-hr-forecast API)
TICK_MINUTES = 5
HORIZON_MINUTES = 60  # a prediction at tick t is about PUB alerts in (t, t + 60 min]
TIERS = ("Low", "Medium", "High")
# Medium and High are cut-offs on a model's chance of a PUB alert within the hour. v10's and v12's are chosen when
# they are fitted: Medium within the rainfall rule's false-alarm rate, High by its flood precision, which no cut-off
# reaches, so High never fires. They can be fixed chances instead (gated.Settings.medium_chance, high_chance): v11 and
# final_1 have Medium from 1% and High from 5%, final_2 Medium from 3% (DEVELOPER_NOTES, the tiers reviewed). The app
# calls the tiers Low, Moderate and Elevated (content.ts, TIER_LABEL); the data keeps the names here.

# Labels start with PUB's Telegram channel (created 26 Oct 2021), most of whose deleted history came back from web
# archive captures. Where posts were never recovered, labels are missing, not zero (gold.label_ticks).
LABELS_START = date(2021, 10, 27)
# The split: train on the first 80% of the recorded label time, test once on the last 20%, with split.EMBARGO_DAYS
# between them. split.temporal_split computed TEST_START on 2026-10-15 from gold.label_ticks: 79.9% of the recorded
# time before the embargo, 20.0% after. It ends at the last midnight of that label snapshot (the channel scraped on
# 13 Oct 2026, 20:37 SGT); later ticks are left for live scorekeeping. Recompute only if the recovered posts change.
TRAIN_START = LABELS_START
TEST_START = date(2025, 12, 20)
SPLIT_END = date(2026, 10, 13)
TEST_SHARE = 0.2
SPLITS = ("train", "embargo", "test", "after_split")

# PUB alert kinds. Telegram archive up to Dec 2025, counted 2026-10-12: 134 warnings on 43 days, 21 flash floods on
# 11 days.
WARNING = "warning"  # "[Risk of Flash Floods]": a drain is at 90% of capacity
FLASH_FLOOD = "flash_flood"  # "[FLASH FLOOD OCCURRED]", or an API Flash Flood Alert with status Alert
ALERT_KINDS = (WARNING, FLASH_FLOOD)

# silver.area_alerts: one row per PUB alert, geocoded to the area its location falls in. An alert seen on both
# Telegram and the API is one row. Not alerts: "Flash flood subsided" posts and NEA's "Heavy rain expected" posts
# (the latter is a candidate feature for the gated model).
AREA_ALERTS = "area_alerts"
AREA_ALERTS_COLUMNS = {
    "area": "NEA forecast area the alert's location falls in",
    "alert_ts": "when PUB issued the alert: the [HH:MM hours] stamp of a Telegram post, else the API timestamp. "
    "Posts can go out minutes after the stamp, so this is the conservative time to measure lead from",
    "kind": f"{WARNING!r} or {FLASH_FLOOD!r}",
    "posted_ts": "when the alert was posted, i.e. when the public could see it",
    "location": "PUB's location text, kept for case studies",
    "lat": "latitude (WGS84) silver.location_gazetteer gives the location: where the alert is, for the features that "
    "look around an area (PUB_ALERTS)",
    "lon": "longitude (WGS84), with lat",
    "source": "'telegram' or 'api'",
    "read_by": "how PUB's text became this alert: 'pattern' (labels/alert_text.py), 'manual' (a hand label in "
    "data/static/telegram_manual_labels.csv) or 'jev' (Jev read a post no pattern reads; ingest/fallback.py)",
}

# gold.area_features: one row per area per tick. Every value must be known at its tick: a reading counts from
# READING_LAG_MINUTES after its timestamp, and not before NEA published it where NEA says when (the yearly CSVs'
# update time). NEA published every 2021-2024 reading more than 5 min after its timestamp, and 4-20% of rain readings
# more than 10 min after (CSV update times, queried 2026-10-15), so a 5-minute lag would read unpublished data.
AREA_FEATURES = "area_features"
READING_LAG_MINUTES = 10
# Rain is the max over nearby gauges, not the mean: flash floods follow the most intense cell, which a mean dilutes.
GAUGE_RADIUS_KM = 5.0
RAIN_WINDOWS_MINUTES = (15, 30, 60)


def rain_column(minutes: int) -> str:
    return f"rain_{minutes}min_mm"


def _rain_doc(minutes: int) -> str:
    return (
        f"largest {minutes}-minute rainfall total among NEA gauges within {GAUGE_RADIUS_KM:g} km of the area's "
        f"forecast point, ending {READING_LAG_MINUTES} min before ts, from the readings NEA had published by ts; "
        "null if none of them had"
    )


# The rain of the hours before the last one: how wet the area already was when the last hour's rain began. With
# rain_60min_mm they make the 3-, 6- and 24-hour totals, split at the last hour so the burst at hand isn't counted
# twice. Per gauge, it is the sum of the gauge's 60-minute totals at the ticks 1, 2, ... hours before, each as NEA had
# published it by its own tick, so never more than a live fetch at the tick had. Given the last hour, more rain before
# it makes an alert *less* likely (gold.model_rows, 23 Oct: at 1-10 mm in the last hour, 2.5 alert rows per thousand
# after two dry hours, 0.26 after 30 mm or more), as PUB alerts when a storm starts, not hours into it. So the models
# may read it either way: no monotone constraint.
PRIOR_RAIN_HOURS = (3, 6, 24)


def prior_rain_column(hours: int) -> str:
    return f"rain_prior_{hours}h_mm"


def _prior_rain_doc(hours: int) -> str:
    return (
        f"the {hours}-hour rainfall total less the last hour's: the largest, among NEA gauges within "
        f"{GAUGE_RADIUS_KM:g} km of the area's forecast point, of a gauge's {max(RAIN_WINDOWS_MINUTES)}-minute totals "
        f"at the {hours - 1} whole hours before ts, each as NEA had published it by its own tick; null if no gauge in "
        "reach reported in every one of those hours"
    )


# The storm so far (24 Oct): how long the rain over the area has gone on, and how much of it the drains haven't taken
# away yet. PUB alerts as a storm starts: in gold.model_rows (24 Oct), with 30 mm or more in the last hour, 27.7 alert
# rows per thousand in a storm's first 30 minutes and 5.4 after four hours; with 10-30 mm, 12.4 and 3.2. The rain windows
# end an hour back, so they can't tell a storm's second hour from its fifth.
# A storm over the area starts at a tick some gauge in reach has rain in its STORM_WINDOW_MINUTES window after
# STORM_GAP_MINUTES without any, and lasts until STORM_GAP_MINUTES pass without. The gap is about how long a drain takes
# to empty: PUB's flash floods subside within the hour (of April 2020's, the environment ministry said they "lasted
# between 10 to 30 minutes"). The age is
# capped at STORM_MAX_MINUTES, so live scoring, which reads a day of readings (features.PRIOR_RAIN_REACH_S), computes it
# as gold does.
STORM_WINDOW_MINUTES = 15
STORM_GAP_MINUTES = 60
STORM_MAX_MINUTES = 12 * 60
STORM_AGE = "storm_age_min"
# The rain surplus: per gauge, a leaky bucket that each reading fills when it arrives and that drains at a constant rate,
# never below empty: the rain that came faster than the drains could take it, still to drain. The rate is the drains'
# capacity as rain over the catchment, which PUB doesn't publish per drain. Its scale: drains built to the standards of
# 2011 on take "up to 45 per cent higher rainfall intensities" than the older ones (a parliamentary reply of the
# environment minister), news reports have 49-56 mm in an hour overwhelming drains (2014 on), and the rainfall rule warns
# from 25 mm in 15 minutes (100 mm/h). So four rates, from a slow store that remembers the whole storm (10 mm/h) to the
# rule's intensity (100 mm/h). As with the prior rain, the models may read the age and the surplus either way: no
# monotone constraint.
DRAIN_RATES_MM_H = (10, 25, 50, 100)


def surplus_column(rate: int) -> str:
    return f"rain_surplus_{rate}mmh_mm"


def _surplus_doc(rate: int) -> str:
    return (
        f"the rain waiting to drain at {rate} mm/h: the largest, among NEA gauges within {GAUGE_RADIUS_KM:g} km of the "
        f"area's forecast point, of a gauge's leaky bucket, which each reading fills at the first tick it counts (from "
        f"{READING_LAG_MINUTES} min after its timestamp, once NEA had published it) and which empties at {rate} mm/h, "
        "never below 0"
    )


RAIN_SURPLUS = tuple(surplus_column(r) for r in DRAIN_RATES_MM_H)
STORM_HISTORY = (STORM_AGE, *RAIN_SURPLUS)

# The drains' response (25 Oct): the surplus's bucket drains at its rate whatever it holds, so it models the water that
# overflows a drain of that capacity, not the drain's water level, which PUB's 90% warnings are about. Here, per gauge,
# a nonlinear reservoir (SWMM's runoff model) stands for the catchment upstream of the drains. Each reading arrives as
# for the surplus; the first DRAIN_INITIAL_LOSS_MM of it after a dry spell wets the ground (SWMM's usual depression
# storage on paved ground is 1.3-2.5 mm) and evaporates again at DRAIN_LOSS_RECOVERY_MM_DAY; the rest is stored, and
# flows into the drains at a rate that rises with what is stored, as an open channel's flow rises with its depth
# (Manning: flow ~ depth^5/3), dS/dt = rain - a * S^(5/3). So the store empties fast while full and slowly near empty, and
# a storm's flow peaks sooner and higher the heavier it is. One parameter sets a: the response time, the reservoir's
# time constant at DRAIN_REFERENCE_MM_H of steady rain, 10 min for a small drain's catchment and 30 min for a canal's
# (DRAIN_RESPONSE_MINUTES). The flow, in mm/h of rain over the catchment, stands for the drain's level: a drain's level
# rises with its flow, and the trees read only the order of a column's values. Flow beyond DRAIN_CAPACITY_MM_H fills an
# overflow store that drains at the capacity left over: the surplus's bucket, fed by the drain's inflow, not the rain.
# The flow is rounded to DRAIN_FLOW_STEP_MM_H, so the long tail of a store near empty reads 0 within hours (14 h after
# the heaviest storm at 30 min), and live scoring, whose stores start empty a day back, computes what gold does.
# And how far each flow is above its highest earlier in the storm (the rise): PUB posts a warning when a drain first
# crosses 90%, so a level the storm has already passed is likely posted. The alert chance may only rise with all three
# (gated.RISING).
DRAIN_RESPONSE_MINUTES = (10, 30)
DRAIN_REFERENCE_MM_H = 50  # the drains' capacity, as for the surplus: about the hourly rain that overwhelms them
DRAIN_EXPONENT = 5 / 3
DRAIN_INITIAL_LOSS_MM = 2.0
DRAIN_LOSS_RECOVERY_MM_DAY = 4.0  # open water evaporates about 4 mm a day in Singapore
DRAIN_CAPACITY_MM_H = 50
DRAIN_FLOW_STEP_MM_H = 0.1


def drain_flow_column(minutes: int) -> str:
    return f"drain_flow_{minutes}min_mmh"


def drain_overflow_column(minutes: int) -> str:
    return f"drain_overflow_{minutes}min_mm"


def drain_rise_column(minutes: int) -> str:
    return f"drain_rise_{minutes}min_mmh"


def _drain_doc(minutes: int) -> str:
    return (
        f"the flow into the drains (mm/h of rain over the catchment, to {DRAIN_FLOW_STEP_MM_H:g}) of a nonlinear "
        f"reservoir with a {minutes}-minute response at {DRAIN_REFERENCE_MM_H} mm/h: the largest, among NEA gauges "
        f"within {GAUGE_RADIUS_KM:g} km of the area's forecast point, of a gauge's store, which each reading fills at the "
        f"first tick it counts, less the first {DRAIN_INITIAL_LOSS_MM:g} mm after a dry spell, and which flows out as "
        f"its content to the power {DRAIN_EXPONENT:.3g}"
    )


def _overflow_doc(minutes: int) -> str:
    return (
        f"the water waiting to enter the drains: the largest, among the same gauges, of a store that the "
        f"{drain_flow_column(minutes)} flow beyond {DRAIN_CAPACITY_MM_H} mm/h fills and that drains at the capacity "
        "left over, never below 0"
    )


def _rise_doc(minutes: int) -> str:
    return (
        f"{drain_flow_column(minutes)} less its highest at the area's earlier ticks of the storm ({STORM_AGE}), 0 "
        f"before the storm's first: above 0 while the storm sets a new high; null outside a storm"
    )


DRAIN_FLOW = tuple(drain_flow_column(m) for m in DRAIN_RESPONSE_MINUTES)
DRAIN_OVERFLOW = tuple(drain_overflow_column(m) for m in DRAIN_RESPONSE_MINUTES)
DRAIN_RISE = tuple(drain_rise_column(m) for m in DRAIN_RESPONSE_MINUTES)
DRAIN_STATE = (*DRAIN_FLOW, *DRAIN_OVERFLOW)
DRAIN_RESPONSE = (*DRAIN_STATE, *DRAIN_RISE)


# Rain beyond the area's own gauges: storms close enough to arrive within the hour. Upwind is within
# UPWIND_HALF_ANGLE_DEG of the direction the island-wide wind blows from; a mean wind under UPWIND_MIN_WIND_KN (calm,
# or stations that disagree) gives no upwind.
NEARBY_RAIN_KM = (GAUGE_RADIUS_KM, 20.0)
NEARBY_RAIN_MINUTES = 30
UPWIND_HALF_ANGLE_DEG = 45.0
UPWIND_MIN_WIND_KN = 2.0
# PUB's own alerts around an area: a PUB alert for another area, at a place near the area's forecast point, often comes
# minutes before the area's own, as a storm moves over neighbouring drains. 23 Oct 2026: 29% of PUB events had one
# posted within 5 km in the hour before, and as a rule alone (every area within 5 km of a posted alert at Medium for 15
# minutes) it flagged 21.4% of the development events at 0.65 false alarms per area-month. A post counts from
# POST_LAG_MINUTES after its posting, as flood-live fetches the channel once a tick and scores ticks up to a tick after
# its fetch. The area's own alerts don't count: its label is about them. Where a post of the window may be lost (the
# label window missing), the count is unknown, not zero.
NEARBY_ALERT_WINDOWS = ((5, 30), (5, 60), (10, 60))  # (km, minutes)
POST_LAG_MINUTES = 5


def nearby_alerts_column(km: int, minutes: int) -> str:
    return f"pub_alerts_{km}km_{minutes}min"


def _nearby_alerts_doc(km: int, minutes: int) -> str:
    return (
        f"PUB alerts (90% warnings and flash floods) posted for other areas at places within {km} km of the area's "
        f"forecast point, in the {minutes} minutes ending {POST_LAG_MINUTES} min before ts; null where a post of that "
        "window may be lost (gold.label_ticks), and in live scoring never"
    )


PUB_ALERTS = tuple(nearby_alerts_column(km, m) for km, m in NEARBY_ALERT_WINDOWS)
# Weather: the mean over NEA's stations of each one's newest reading in the WEATHER_MINUTES ending
# READING_LAG_MINUTES before the tick, published by the tick. Island-wide, as 19 stations can't give 47 areas their own.
WEATHER_MINUTES = 5
# silver.weather_1min's variables (pipeline/utilities/silver_sources.py WEATHER_VARIABLES, which a test keeps equal)
WEATHER_READINGS = ("air_temp_c", "rel_humidity_pct", "wind_speed_kn", "wind_dir_deg")
# The dew point, by the Magnus formula (Alduchov and Eskridge 1996), from each station's own temperature and humidity
# before the mean over stations. Relative humidity moves with the temperature (it falls as the day warms with no change
# in the air's moisture), so the models read the moisture as the dew point beside the temperature (23 Oct). All 38
# stations report both (silver.weather_1min, 23 Oct).
MAGNUS_A, MAGNUS_B_C = 17.625, 243.04


def dew_point_c(air_temp_c, rel_humidity_pct):
    """The Magnus dew point (deg C) of numbers or numpy arrays; features.tick_weather computes the same in Spark."""
    import numpy as np

    t, rh = np.asarray(air_temp_c, dtype=float), np.asarray(rel_humidity_pct, dtype=float)
    gamma = np.log(rh / 100) + MAGNUS_A * t / (MAGNUS_B_C + t)
    return MAGNUS_B_C * gamma / (MAGNUS_A - gamma)
# Lightning: NEA lists strikes in a record every 2 minutes, published about 2 minutes after it closes, from
# LIGHTNING_START. A tick's window is known only when every record of it exists.
LIGHTNING_START = date(2025, 2, 1)
LIGHTNING_MINUTES = 30
LIGHTNING_RECORD_MINUTES = 2
LIGHTNING_RECORDS_PER_WINDOW = LIGHTNING_MINUTES // LIGHTNING_RECORD_MINUTES
LIGHTNING_RADII_KM = (10, 30)


def lightning_column(km: int) -> str:
    return f"lightning_{km}km_{LIGHTNING_MINUTES}min"


def _lightning_doc(km: int) -> str:
    return (
        f"lightning strikes NEA observed within {km} km of the area's forecast point in the {LIGHTNING_MINUTES} "
        f"minutes before ts, as published by ts; null before {LIGHTNING_START.isoformat()} and where NEA's "
        f"{LIGHTNING_RECORD_MINUTES}-minute records of those minutes are incomplete"
    )


# NEA's 2-hour forecast, as a live fetch at the tick would have had it: the newest one published (updated_at) by ts,
# while its valid period lasts. rain_class orders the categories (pipeline/utilities/forecast_categories.py).
# The tide: a harmonic model fitted on UHSLC's hourly sea level at Tanjong Pagar (1984-2023), checked against NEA's
# high and low waters (gold.tide_datum). The models read the prediction, not the gauge: no gauge reading is available
# live, and the 2024 onward record has none, so training on observed surges would teach the models an input they never
# see when scoring. gold.tide_levels keeps the gauge-filled series for analysis.
TIDE_START = date(
    2021, 1, 1
)  # the first SGT day gold.tide_levels fills, with the first rain readings
# NEA re-published every 2-hour forecast issued from 1 Jan to 8 May 2025 07:30 SGT in one batch, on 8 May 2025 between
# 11:00 and 13:00 SGT. Their updated_at says when NEA rebuilt its archive, not when a live fetch got them, so every tick
# of those four months read as having no forecast: 124 training days, with 7 of training's 36 flash floods (21 Oct).
# Their issue times are NEA's own, so a forecast published in that batch counts from
# REPUBLISHED_FORECAST_DELAY_MINUTES after its issue: NEA's API published 99% of its other forecasts within 9.6 minutes
# of issue (silver.forecasts_2h, 2025-26), so the models read them no earlier than a live fetch would have. Forecasts
# NEA published late for other reasons (its own outages) keep their times.
FORECASTS_REPUBLISHED_SGT = (datetime(2025, 5, 8, 11, 0), datetime(2025, 5, 8, 13, 0))
REPUBLISHED_FORECAST_DELAY_MINUTES = 10

RAIN_CLASSES = {
    0: "no rain",
    1: "light rain or showers",
    2: "moderate rain or showers",
    3: "heavy rain or showers",
    4: "thundery showers",
    5: "heavy thundery showers",
}
# The rain classes in fewer groups (24 Oct), each a list of the group of class 0, 1, ... 5. NEA's 24 wordings are already
# 6 classes, and the models read the class as one number, so they can merge neighbours on their own; but class 3 is rare
# (training, 24 Oct: 15 alert rows on 4 area-days, against 221 for class 4). The three groups are the meteorologist's
# (dry, rain, thunder); the four keep thundery and heavy thundery apart, which alert at 4.0 and 11.4 rows per thousand.
FORECAST_GROUPINGS = {
    "forecast_rain_group3": (0, 1, 1, 1, 2, 2),
    "forecast_rain_group4": (0, 0, 1, 1, 2, 3),
}
FORECAST_GROUP_NAMES = {
    "forecast_rain_group3": "0 no rain, 1 rain or showers (classes 1-3), 2 thundery (4-5)",
    "forecast_rain_group4": "0 no rain or light (classes 0-1), 1 moderate or heavy (2-3), 2 thundery showers (4), "
    "3 heavy thundery showers (5)",
}
FORECAST_GROUPED = tuple(FORECAST_GROUPINGS)

# The ground (25 Oct): what each area's land is like, from the national map (gold.area_terrain): one value per area, the
# same at every tick. The area prior already gives each area its own alert rate (gated.py), so the ground can only add
# where it changes what the weather does: a high tide on low land by the sea, a downpour on the hills above a valley.
# SLA's contours are every 20 m (Contour_250K), so the land above HIGH_GROUND_M is the hills, and the national map has
# no heights finer than that (its vertical control benchmarks carry none). Each column is measured over the area's land,
# the Voronoi cell of its label point where its alerts are placed, as gold.area_land_use measures land use.
AREA_TERRAIN = "area_terrain"
HIGH_GROUND_M = 20
MAP_LINE = ("high_ground_km", "relief_m", "road_share")  # from the map's lines: contours and roads
MAP_POLYGON = ("coast_km", "waterway_km", "green_share")  # from its polygons: coast, waterways, parks
MAP_FEATURES = (*MAP_LINE, *MAP_POLYGON)

AREA_FEATURES_COLUMNS = {
    "area": "NEA forecast area",
    "ts": "tick, on a 5-minute boundary",
    **{rain_column(w): _rain_doc(w) for w in RAIN_WINDOWS_MINUTES},
    **{prior_rain_column(h): _prior_rain_doc(h) for h in PRIOR_RAIN_HOURS},
    STORM_AGE: f"minutes since the storm over the area began: the first tick a gauge within {GAUGE_RADIUS_KM:g} km had "
    f"rain in its {STORM_WINDOW_MINUTES}-minute window after {STORM_GAP_MINUTES} min without, at most "
    f"{STORM_MAX_MINUTES}; null when none had rain in the last {STORM_GAP_MINUTES} min",
    **{surplus_column(r): _surplus_doc(r) for r in DRAIN_RATES_MM_H},
    **{drain_flow_column(m): _drain_doc(m) for m in DRAIN_RESPONSE_MINUTES},
    **{drain_overflow_column(m): _overflow_doc(m) for m in DRAIN_RESPONSE_MINUTES},
    **{drain_rise_column(m): _rise_doc(m) for m in DRAIN_RESPONSE_MINUTES},
    "rain_gauges": f"gauges within {GAUGE_RADIUS_KM:g} km whose readings count in the {max(RAIN_WINDOWS_MINUTES)}-"
    "minute window",
    "forecast_rain_class": "NEA's 2-hour forecast for the area, the newest published by ts and still valid, as its "
    "rain class: "
    + ", ".join(f"{k} {v}" for k, v in RAIN_CLASSES.items())
    + "; null if none was valid",
    **{
        name: f"forecast_rain_class in fewer groups: {FORECAST_GROUP_NAMES[name]}; null with it"
        for name in FORECAST_GROUPED
    },
    "forecast_age_min": "minutes from that forecast's publication to ts; null with it",
    "rain_nearby_30min_mm": f"largest {NEARBY_RAIN_MINUTES}-minute rainfall total among NEA gauges "
    f"{NEARBY_RAIN_KM[0]:g}-{NEARBY_RAIN_KM[1]:g} km from the area's forecast point, from the readings published by "
    "ts as for the rain columns: rain close enough to arrive within the hour; 0 if none of those gauges rained",
    "rain_upwind_30min_mm": f"the same, among those gauges within {UPWIND_HALF_ANGLE_DEG:g} degrees of the direction "
    f"the wind blows from (wind_u_kn, wind_v_kn); null when the wind is unknown or under {UPWIND_MIN_WIND_KN:g} knots",
    **{nearby_alerts_column(km, m): _nearby_alerts_doc(km, m) for km, m in NEARBY_ALERT_WINDOWS},
    "air_temp_c": "mean air temperature (deg C) over NEA's weather stations, each station's newest reading in the "
    f"{WEATHER_MINUTES} minutes ending {READING_LAG_MINUTES} min before ts that NEA had published by ts; null if no "
    "station had one",
    "rel_humidity_pct": "mean relative humidity (%), read as air_temp_c",
    "dew_point_c": "mean dew point (deg C) over the stations, each station's from its own air temperature and "
    "relative humidity (Magnus formula) before the mean, read as air_temp_c",
    "wind_u_kn": "mean wind over the stations, read as air_temp_c: the eastward component (knots) of NEA's 10-minute "
    "means",
    "wind_v_kn": "its northward component (knots)",
    **{lightning_column(km): _lightning_doc(km) for km in LIGHTNING_RADII_KM},
    "tide_height_m": "predicted tide at Tanjong Pagar at ts, metres above Chart Datum (gold.tide_levels.predicted_m): "
    "the astronomical tide, known in advance, so it is the same live and in replays; the same for every area",
    "minutes_to_high_tide": "minutes from ts to the next predicted high water at Tanjong Pagar",
    "high_ground_km": f"mean distance (km) from the area's land to the nearest contour of {HIGH_GROUND_M} m or more on "
    "the national map: how far the hills are (gold.area_terrain); the same at every tick",
    "relief_m": f"the highest contour within {GAUGE_RADIUS_KM:g} km of the area's forecast point (metres, a multiple "
    f"of 20), 0 where none of {HIGH_GROUND_M} m or more is: how high the hills around it rise",
    "road_share": "share of the area's land cells (H3, about 50 m across) that an expressway, slip road or major road "
    "of the national map crosses",
    "coast_km": "mean distance (km) from the area's land to the sea, the edge of Singapore's coastal outlines",
    "waterway_km": "mean distance (km) from the area's land to the nearest canal or river of the national map",
    "green_share": "share of the area's land in the national map's parks and nature reserves",
}
# gold.area_features columns the models don't read yet: lightning (16 Oct). To feed it to them, drop it from here and
# give it its monotone constraint and its reason in gated.py.
HELD_BACK_FEATURES = tuple(lightning_column(km) for km in LIGHTNING_RADII_KM)
# The columns a model may read, each as it stood at the tick. Every model also reads the SGT hour, and leaves out what
# its feature set does (BASE_FEATURES or EXTENDED_FEATURES, below).
MODEL_FEATURES = tuple(
    c for c in AREA_FEATURES_COLUMNS if c not in ("area", "ts", *HELD_BACK_FEATURES)
)
# Of those, the ones a tick needs to be modelled or rated: rain near the area and the forecast (gold.feature_ticks).
# The rest come from sources that can be missing at a tick (weather stations have outages, and their 2025-26 days
# arrive with the backfill; a calm has no upwind), so they stay null there, and LightGBM reads a null as missing. The
# tide is predicted for every tick, so it is optional only in name.
REQUIRED_FEATURES = (
    *(rain_column(w) for w in RAIN_WINDOWS_MINUTES),
    "rain_gauges",
    "forecast_rain_class",
    "forecast_age_min",
)
OPTIONAL_FEATURES = tuple(c for c in MODEL_FEATURES if c not in REQUIRED_FEATURES)
PRIOR_RAIN = tuple(prior_rain_column(h) for h in PRIOR_RAIN_HOURS)
# The two feature sets a model is fitted on ('base' and 'extended' in a flood-train spec), both within MODEL_FEATURES,
# which gold provides. BASE_FEATURES is what the gated pair has read since 16 Oct (v5): the default, as the tree models
# did best on it (23 Oct, out of fold: the pair flagged 30.1% of events before PUB on it, 25.5% on the extended set).
# EXTENDED_FEATURES swaps relative humidity, which is mostly the temperature again, for the dew point, and adds the prior
# rain; it helped the logistic model by a hair (ARCHITECTURE.md).
# Neither reads PUB's alerts nearby (PUB_ALERTS, 23 Oct), the storm so far (STORM_HISTORY) or the forecast groups
# (FORECAST_GROUPED, 24 Oct), the ground (MAP_FEATURES) or the drains' response (DRAIN_RESPONSE, 25 Oct): a spec adds
# them with add= (add=pub_alerts, add=storm, add=map_line, add=drain, ...; train.py).
OUTSIDE_SETS = (*PUB_ALERTS, *STORM_HISTORY, *DRAIN_RESPONSE, *FORECAST_GROUPED, *MAP_FEATURES)
BASE_FEATURES = tuple(c for c in MODEL_FEATURES if c not in ("dew_point_c", *PRIOR_RAIN, *OUTSIDE_SETS))
EXTENDED_FEATURES = tuple(c for c in MODEL_FEATURES if c not in ("rel_humidity_pct", *OUTSIDE_SETS))
# Model features by source, so flood-train can fit a model without a group and measure what it adds.
FEATURE_GROUPS = {
    "rain": (*(rain_column(w) for w in RAIN_WINDOWS_MINUTES), "rain_gauges"),
    "antecedent": PRIOR_RAIN,
    "storm": (STORM_AGE,),
    "surplus": RAIN_SURPLUS,
    "drain": DRAIN_STATE,
    "rise": DRAIN_RISE,
    "forecast": ("forecast_rain_class", "forecast_age_min"),
    "forecast_groups": FORECAST_GROUPED,
    "nearby": ("rain_nearby_30min_mm", "rain_upwind_30min_mm"),
    "pub_alerts": PUB_ALERTS,
    "weather": ("air_temp_c", "rel_humidity_pct", "dew_point_c", "wind_u_kn", "wind_v_kn"),
    "tide": ("tide_height_m", "minutes_to_high_tide"),
    "map_line": MAP_LINE,
    "map_polygon": MAP_POLYGON,
}

# Whether a tick's label is known. Its label reads PUB alerts stamped in (ts, ts + HORIZON]; PUB posts an alert at
# most 35 min after its stamp (Telegram 2021-2026; rounded up here), and the post parser accepts a stamp up to
# STAMP_AHEAD_MINUTES after its posting (alert_text.CLOCK_SKEW). So the label is complete only if every post from
# (ts - STAMP_AHEAD, ts + HORIZON + POST_DELAY_MAX] was recovered. Telegram post ids run without gaps, so two posts
# with consecutive ids prove nothing was posted between them; an id gap leaves the whole span between its neighbours
# unknown, however few posts are missing.
POST_DELAY_MAX_MINUTES = 60
STAMP_AHEAD_MINUTES = 10
RECORDED, MISSING = "recorded", "missing"
POSTS_NOT_RECOVERED = "posts not recovered"  # an id gap: posts PUB deleted before the web archive captured them
# a post was recovered but can't be read: no pattern or hand label reads it, and Jev failed on it (its endpoint missing
# included), so it may hold an alert nobody knows of
UNREAD_POST = "post not read"
UNPLACED_ALERT = (
    "unplaced alert"  # an alert came, but its location isn't placed in an area yet
)
MISSING_REASONS = (POSTS_NOT_RECOVERED, UNREAD_POST, UNPLACED_ALERT)

# gold.label_ticks: one row per tick from LABELS_START to the last complete midnight. The same for every area.
LABEL_TICKS = "label_ticks"
LABEL_TICKS_COLUMNS = {
    "ts": "tick, on a 5-minute boundary",
    "label_window": f"{RECORDED!r} if every post that could carry an alert for (ts, ts + {HORIZON_MINUTES} min] "
    f"was recovered and read and every alert in it is placed, else {MISSING!r}",
    "missing_reason": f"why the window is missing, first that applies: {POSTS_NOT_RECOVERED!r}, {UNREAD_POST!r} or "
    f"{UNPLACED_ALERT!r}; null when recorded",
    "split": f"{' / '.join(SPLITS)}: train before the embargo, test from TEST_START to SPLIT_END",
}

# gold.area_labels: one row per area per tick of gold.label_ticks. The labels scoreboard.tick_labels computes, with
# missing windows kept apart from quiet ones.
AREA_LABELS = "area_labels"
ALERT, NO_ALERT = "alert", "no_alert"
AREA_LABELS_COLUMNS = {
    "area": "NEA forecast area",
    "ts": "tick, on a 5-minute boundary",
    "alert_status": f"{ALERT!r}: a PUB alert came in the area within (ts, ts + {HORIZON_MINUTES} min]; "
    f"{NO_ALERT!r}: none came, and the window was recorded; {MISSING!r}: none was recorded, but the window wasn't",
    "alert_label": "1, 0 or null, as alert_status: the label for the Medium tier",
    "flood_label": f"1 if a flash flood came in the area within (ts, ts + {HORIZON_MINUTES} min], 0 if none came "
    "and the window was recorded, else null: the label for High",
    "missing_reason": LABEL_TICKS_COLUMNS["missing_reason"],
    "split": LABEL_TICKS_COLUMNS["split"],
}

# Whether a tick's features are known. NEA's rain feed and forecasts have outages (1-9 Jan 2023: no readings at all),
# and the API backfill lands later years over days. An outage leaves no row, which must not read as a dry tick: a
# tick's feature window is recorded only if the gauge network reported at every reading timestamp its rain windows
# read, and a forecast was valid. Outages are island-wide, so this is per tick like gold.label_ticks.
# a reading timestamp in the tick's hour with no gauge reporting; no 2-hour forecast valid at the tick
RAIN_GAP, FORECAST_GAP = "rain feed gap", "forecast gap"
FEATURE_MISSING_REASONS = (RAIN_GAP, FORECAST_GAP)
FEATURE_TICKS = "feature_ticks"
FEATURE_TICKS_COLUMNS = {
    "ts": "tick, on a 5-minute boundary",
    "rain_stamps": f"reading timestamps the tick's {max(RAIN_WINDOWS_MINUTES)}-minute rain window reads at which some "
    f"gauge reported, of {max(RAIN_WINDOWS_MINUTES) // TICK_MINUTES}",
    "forecast_areas": "areas with a 2-hour forecast valid at the tick",
    "feature_window": f"{RECORDED!r} if every reading timestamp of the rain window came in and a forecast was valid, "
    f"else {MISSING!r}",
    "missing_reason": f"why the window is missing: {RAIN_GAP!r} (takes precedence) or {FORECAST_GAP!r}; null when "
    "recorded",
}

# gold.model_rows: the area-ticks models learn from (split 'train') and are tested on (split 'test'): every row of
# gold.area_features joined to gold.area_labels where the label window and the feature window are both recorded and
# no required feature is null. A missing interval never reaches a model, not even one that saw an alert: keeping those
# would keep a missing window's alerts but none of its quiet ticks, and inflate the alert rate wherever posts were lost.
MODEL_ROWS = "model_rows"
MODEL_ROWS_COLUMNS = {
    "area": "NEA forecast area",
    "ts": "tick, on a 5-minute boundary",
    **{c: AREA_FEATURES_COLUMNS[c] for c in MODEL_FEATURES},
    "alert_label": AREA_LABELS_COLUMNS["alert_label"],
    "flood_label": AREA_LABELS_COLUMNS["flood_label"],
    "split": "'train' or 'test' (contract.py)",
}

# ml.predictions: one row per model, area and tick. The app, the dashboard and the scoreboard read it.
PREDICTIONS = "predictions"
PREDICTIONS_COLUMNS = {
    "model": "model name and parameters or version",
    "area": "NEA forecast area",
    "ts": "tick the prediction was made at",
    "tier": " / ".join(TIERS),
    "p_alert": "calibrated probability of a PUB alert in the area within the next hour; null for the rule",
    "reason": "plain-English trigger shown in the app",
}

# The registered model, in Unity Catalog as <catalog>.<ml schema>.floodsense: the ordinal model from 23 Oct
# (ordinal.py), the gated LightGBM pair before (gated.py; v1 to v5).
REGISTERED_MODEL = "floodsense"
# UC aliases: the live model, and the newest one trained
CHAMPION, CHALLENGER = "champion", "challenger"


def version_name(model: str, version, aliases=()) -> str:
    """What ml.predictions and the app call a registered version: the alias the user named it by (final_1, 25 Oct),
    when it has one besides CHAMPION and CHALLENGER, else '<model> v<version>'."""
    named = sorted(set(aliases or ()) - {CHAMPION, CHALLENGER})
    return named[0] if named else f"{model} v{version}"
