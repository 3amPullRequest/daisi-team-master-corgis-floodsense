"""What every FloodSense model reads and writes. Build silver and gold backwards from here.

The unit of prediction is one NEA forecast area at one 5-minute tick: how likely is a PUB alert in that area within the
next hour? The rainfall-rule baseline and the gated LightGBM read the same tables and are judged by the same scoreboard
(scoreboard.py). The gated model only reads more feature columns: it adds columns to `area_features`, never a new
grain, label or scoring rule.

Timestamps are SGT; naive ones are read as SGT.
"""

from __future__ import annotations

from datetime import date

N_AREAS = 47  # NEA 2-hour forecast areas (`area_metadata` of the two-hr-forecast API)
TICK_MINUTES = 5
HORIZON_MINUTES = 60  # a prediction at tick t is about PUB alerts in (t, t + 60 min]
TIERS = ("Low", "Medium", "High")

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
    "source": "'telegram' or 'api'",
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


# NEA's 2-hour forecast, as a live fetch at the tick would have had it: the newest one published (updated_at) by ts,
# while its valid period lasts. rain_class orders the categories (pipeline/utilities/forecast_categories.py).
# The tide: a harmonic model fitted on UHSLC's hourly sea level at Tanjong Pagar (1984-2023), checked against NEA's
# high and low waters (gold.tide_datum). The models read the prediction, not the gauge: no gauge reading is available
# live, and the 2024 onward record has none, so training on observed surges would teach the models an input they never
# see when scoring. gold.tide_levels keeps the gauge-filled series for analysis.
TIDE_START = date(
    2021, 1, 1
)  # the first SGT day gold.tide_levels fills, with the first rain readings

RAIN_CLASSES = {
    0: "no rain",
    1: "light rain or showers",
    2: "moderate rain or showers",
    3: "heavy rain or showers",
    4: "thundery showers",
    5: "heavy thundery showers",
}

AREA_FEATURES_COLUMNS = {
    "area": "NEA forecast area",
    "ts": "tick, on a 5-minute boundary",
    **{rain_column(w): _rain_doc(w) for w in RAIN_WINDOWS_MINUTES},
    "rain_gauges": f"gauges within {GAUGE_RADIUS_KM:g} km whose readings count in the {max(RAIN_WINDOWS_MINUTES)}-"
    "minute window",
    "forecast_rain_class": "NEA's 2-hour forecast for the area, the newest published by ts and still valid, as its "
    "rain class: "
    + ", ".join(f"{k} {v}" for k, v in RAIN_CLASSES.items())
    + "; null if none was valid",
    "forecast_age_min": "minutes from that forecast's publication to ts; null with it",
    "tide_height_m": "predicted tide at Tanjong Pagar at ts, metres above Chart Datum (gold.tide_levels.predicted_m): "
    "the astronomical tide, known in advance, so it is the same live and in replays; the same for every area",
    "minutes_to_high_tide": "minutes from ts to the next predicted high water at Tanjong Pagar",
}
# The columns a model reads, all of them known at the tick. The gated pair also reads the area and the SGT hour.
MODEL_FEATURES = tuple(c for c in AREA_FEATURES_COLUMNS if c not in ("area", "ts"))
# Model features by source, so flood-train can fit the pair without a group and measure what it adds.
FEATURE_GROUPS = {
    "rain": (*(rain_column(w) for w in RAIN_WINDOWS_MINUTES), "rain_gauges"),
    "forecast": ("forecast_rain_class", "forecast_age_min"),
    "tide": ("tide_height_m", "minutes_to_high_tide"),
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
UNPLACED_ALERT = (
    "unplaced alert"  # an alert came, but its location isn't placed in an area yet
)
MISSING_REASONS = (POSTS_NOT_RECOVERED, UNPLACED_ALERT)

# gold.label_ticks: one row per tick from LABELS_START to the last complete midnight. The same for every area.
LABEL_TICKS = "label_ticks"
LABEL_TICKS_COLUMNS = {
    "ts": "tick, on a 5-minute boundary",
    "label_window": f"{RECORDED!r} if every post that could carry an alert for (ts, ts + {HORIZON_MINUTES} min] "
    f"was recovered and every alert in it is placed, else {MISSING!r}",
    "missing_reason": f"why the window is missing: {POSTS_NOT_RECOVERED!r} (takes precedence) or "
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
# no model feature is null. A missing interval never reaches a model, not even one that saw an alert: keeping those
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

# The gated LightGBM pair, registered in Unity Catalog as <catalog>.<ml schema>.floodsense.
REGISTERED_MODEL = "floodsense"
# UC aliases: the live model, and the newest one trained
CHAMPION, CHALLENGER = "champion", "challenger"
