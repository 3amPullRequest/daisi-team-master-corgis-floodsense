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

# The Round 1 split: fit on alerts up to December 2025, test on every 2026 alert to date. Usable Telegram alerts
# start in March 2025 (DEVELOPER_NOTES.md), so training starts there.
TRAIN_START = date(2025, 3, 1)
TEST_START = date(2026, 1, 1)

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

# gold.area_features: one row per area per tick. Every value must be known at its tick: a reading counts only once
# the live fetch would have had it. Until that lag is measured from bronze live ticks (fetched_at minus the newest
# reading's timestamp), assume one tick.
AREA_FEATURES = "area_features"
READING_LAG_MINUTES = 5
# Rain is the max over nearby gauges, not the mean: flash floods follow the most intense cell, which a mean dilutes.
GAUGE_RADIUS_KM = 5.0
RAIN_WINDOWS_MINUTES = (15, 30, 60)


def rain_column(minutes: int) -> str:
    return f"rain_{minutes}min_mm"


def _rain_doc(minutes: int) -> str:
    return (
        f"largest {minutes}-minute rainfall total among NEA gauges within {GAUGE_RADIUS_KM:g} km of the area's "
        f"forecast point, ending {READING_LAG_MINUTES} min before ts; null if none of them reported"
    )


AREA_FEATURES_COLUMNS = {
    "area": "NEA forecast area",
    "ts": "tick, on a 5-minute boundary",
    **{rain_column(w): _rain_doc(w) for w in RAIN_WINDOWS_MINUTES},
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
