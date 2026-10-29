"""Publish the live model's scoreboard as tables, for the agencies' AI/BI dashboard and Genie space.

The flood-scoreboard job runs it whenever @champion moves (resources/scoreboard.job.yml):

    databricks bundle run scoreboard -t dev -p flood_dev

Two periods, each judged by scoreboard.py exactly as training judges a model:

- development (27 Oct 2021 to 19 Dec 2025), out of fold: every storm was scored by a version that never trained on it.
  These are the champion's training run's own numbers (its oof_*, oof_high_* and rule_dev_* metrics, oof_events.json
  and oof_floods.json) read back from MLflow, with the rule's lead before each flood from the version's verify run;
- after training (from contract.TEST_START to the newest tick gold has labels and features for): storms the live model
  never trained on, scored here on every area-tick of gold.area_features by the champion and by the rule its training
  run fitted. Its 2026 floods were looked at while versions were chosen (verify's floods.json), so it is the record
  after training rather than an untouched test.

Every Moderate or Elevated rating is a warning for the next hour, so a warning runs from its first rating to an hour
after its last, and a lead time runs from that first rating to PUB's alert (scoreboard.py). The app's storm replays,
its live map and its Telegram posts count warnings the same way (broadcast.py).

It replaces four tables in the ml schema, their columns described for Genie (the *_COLUMNS below):

- scoreboard: a row per period, model and warning level (FloodSense from Moderate, FloodSense at Elevated, the rule),
  with the Round 1 measures and their 95% intervals;
- scoreboard_floods: a row per flash flood, with PUB's own warning, FloodSense's lead and the rule's;
- scoreboard_events: a row per PUB event (alerts in one area at most an hour apart) per period, model and level;
- scoreboard_warnings: a row per warning after training, per model and level, and whether a PUB alert followed;

and puts three views beside them, so the dashboard and Genie read one schema: pub_alerts (silver.area_alerts),
flood_prone_trend (gold) and live_risk (the newest tick of ml.predictions). Times are SGT wall-clock times
(TIMESTAMP_NTZ, whatever the reader's session time zone), and tiers are named as the app shows them: Low, Moderate
(Medium in the data) and Elevated (High).
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract as C
from . import floods
from .scoreboard import TICK, alert_events, score

log = logging.getLogger(__name__)

DEVELOPMENT, AFTER_TRAINING = "development", "after_training"
FLOODSENSE, RULE = "FloodSense", "Rainfall rule"
# what each row counts as a warning
MODERATE, ELEVATED, HEAVY_RAIN = "Moderate or above", "Elevated", "Heavy rain"
SHOWN_TIER = {"Low": "Low", "Medium": "Moderate", "High": "Elevated"}  # the app's names (frontend lib/content.ts)
EVENT_KIND = {"alert": "pub_alert", "flood": "flash_flood"}
OUT_OF_FOLD = "out of fold"
# the training run's metrics for each development row: FloodSense's at Medium and up, at High alone, and the rule's
DEVELOPMENT_ROWS = ((FLOODSENSE, MODERATE, "oof_"), (FLOODSENSE, ELEVATED, "oof_high_"), (RULE, HEAVY_RAIN, "rule_dev_"))
FOLLOWED = {"oof_": "oof_medium_warning_alert_share", "oof_high_": "oof_high_warning_alert_share"}

PERIOD_DOC = (
    "'development' (27 Oct 2021 to 19 Dec 2025, out of fold: each storm scored by a FloodSense version that never "
    "trained on it) or 'after_training' (from 20 Dec 2025: storms the live model never trained on)"
)
AREA_DOC = "NEA 2-hour forecast area (one of 47), where PUB's alert location falls"
IDENTITY = {
    "period": ("STRING", PERIOD_DOC),
    "model": ("STRING", "'FloodSense', or 'Rainfall rule': the simple rule FloodSense has to beat"),
    "model_version": ("STRING", "the FloodSense version (as ml.predictions names it), or the rule's definition"),
    "warning_level": (
        "STRING",
        "what counts as a warning: 'Moderate or above' or 'Elevated' for FloodSense, 'Heavy rain' for the rule",
    ),
}
SCOREBOARD_COLUMNS = {
    **IDENTITY,
    "period_start": ("DATE", "first day of the period (SGT)"),
    "period_end": ("DATE", "last day of the period (SGT)"),
    "pub_alert_events": (
        "BIGINT",
        "PUB alert events judged: PUB alerts in one area at most an hour apart are one event, roughly one storm",
    ),
    "pub_alert_days": ("BIGINT", "storm days those events fell on"),
    "alerts_warned_before_pub": (
        "DOUBLE",
        "share of PUB alert events (0 to 1) with a warning in force before PUB's first alert",
    ),
    "alerts_warned_before_pub_lo": ("DOUBLE", "95% interval for alerts_warned_before_pub, low end (Wilson)"),
    "alerts_warned_before_pub_hi": ("DOUBLE", "95% interval for alerts_warned_before_pub, high end (Wilson)"),
    "alert_mean_lead_min": (
        "DOUBLE",
        "mean minutes from the start of the warning in force to PUB's first alert, a missed event counting 0 (each "
        "Moderate or Elevated rating is a warning for the next hour, so a warning starts at its first rating)",
    ),
    "alert_mean_lead_min_lo": ("DOUBLE", "95% interval for alert_mean_lead_min, low end (bootstrap over storm days)"),
    "alert_mean_lead_min_hi": ("DOUBLE", "95% interval for alert_mean_lead_min, high end (bootstrap over storm days)"),
    "flash_floods": ("BIGINT", "flash floods judged"),
    "floods_warned_before_pub": (
        "DOUBLE",
        "share of flash floods (0 to 1) with a warning in force before PUB's first alert of the storm",
    ),
    "floods_warned_before_pub_lo": ("DOUBLE", "95% interval for floods_warned_before_pub, low end (Wilson)"),
    "floods_warned_before_pub_hi": ("DOUBLE", "95% interval for floods_warned_before_pub, high end (Wilson)"),
    "flood_mean_lead_min": (
        "DOUBLE",
        "mean minutes from the warning's start to the flash flood, a missed flood counting 0",
    ),
    "pub_flood_warning_min": (
        "DOUBLE",
        "PUB's own mean warning before a flash flood: minutes from its first alert of the storm to the flood",
    ),
    "public_flood_warning_min": (
        "DOUBLE",
        "mean warning the public got before a flash flood from whichever came first, this warning or PUB's alert",
    ),
    "public_flood_warning_min_lo": ("DOUBLE", "95% interval for public_flood_warning_min, low end (bootstrap)"),
    "public_flood_warning_min_hi": ("DOUBLE", "95% interval for public_flood_warning_min, high end (bootstrap)"),
    "warnings": (
        "BIGINT",
        "warnings judged: warned 5-minute ticks in one area at most an hour apart are one warning",
    ),
    "warnings_followed_by_alert": (
        "DOUBLE",
        "share of warnings (0 to 1) that a PUB alert in the area came within; the rest are false alarms",
    ),
    "false_alarms_per_area_month": ("DOUBLE", "warnings no PUB alert came within, per area per month"),
    "warned_hours_per_area_month": ("DOUBLE", "hours under a warning per area per month"),
}
# Scoreboard.metrics' names (and the training run's, after its prefix) for the scoreboard's numbers
SUMMARY = {
    "alert_events": "pub_alert_events",
    "alert_days": "pub_alert_days",
    "alert_flagged_before_pub": "alerts_warned_before_pub",
    "alert_flagged_before_pub_lo": "alerts_warned_before_pub_lo",
    "alert_flagged_before_pub_hi": "alerts_warned_before_pub_hi",
    "alert_mean_lead_min": "alert_mean_lead_min",
    # an alert event is timed at PUB's first alert, so the public's warning before it is the lead itself
    "alert_mean_warning_min_lo": "alert_mean_lead_min_lo",
    "alert_mean_warning_min_hi": "alert_mean_lead_min_hi",
    "flood_events": "flash_floods",
    "flood_flagged_before_pub": "floods_warned_before_pub",
    "flood_flagged_before_pub_lo": "floods_warned_before_pub_lo",
    "flood_flagged_before_pub_hi": "floods_warned_before_pub_hi",
    "flood_mean_lead_min": "flood_mean_lead_min",
    "flood_mean_pub_warning_min": "pub_flood_warning_min",
    "flood_mean_warning_min": "public_flood_warning_min",
    "flood_mean_warning_min_lo": "public_flood_warning_min_lo",
    "flood_mean_warning_min_hi": "public_flood_warning_min_hi",
    "warnings": "warnings",
    "false_alarms_per_area_month": "false_alarms_per_area_month",
    "warned_hours_per_area_month": "warned_hours_per_area_month",
}
SHARES = tuple(c for c in SUMMARY.values() if c.startswith(("alerts_warned_before_pub", "floods_warned_before_pub")))
FLOODS_COLUMNS = {
    "flood_time": ("TIMESTAMP_NTZ", "when PUB reported the flash flood (SGT)"),
    "area": ("STRING", AREA_DOC),
    "period": ("STRING", PERIOD_DOC),
    "judged": (
        "BOOLEAN",
        "whether the flood counts in the scoreboard: false where PUB posts around it were lost or the model's inputs "
        "were missing",
    ),
    "pub_first_alert": ("TIMESTAMP_NTZ", "PUB's first alert of that storm in the area (SGT)"),
    "pub_warning_min": ("DOUBLE", "PUB's own warning: minutes from its first alert of the storm to the flood"),
    "floodsense_warned_from": (
        "TIMESTAMP_NTZ",
        "start of the FloodSense warning (Moderate or above) in force at the flood (SGT); null if none",
    ),
    "floodsense_lead_min": (
        "DOUBLE",
        "minutes from that warning's start to the flood; null if no FloodSense warning was in force",
    ),
    "floodsense_before_pub": ("BOOLEAN", "whether FloodSense's warning started before PUB's first alert"),
    "public_warning_min": (
        "DOUBLE",
        "the warning the public got with FloodSense: minutes before the flood from whichever came first, FloodSense "
        "or PUB",
    ),
    "floodsense_tier_at_flood": (
        "STRING",
        "FloodSense's tier at the last 5-minute tick before the flood: Low, Moderate or Elevated",
    ),
    "floodsense_chance_at_flood": (
        "DOUBLE",
        "FloodSense's chance (0 to 1) of a PUB alert in the area within the hour, at that tick",
    ),
    "floodsense_peak_tier": ("STRING", "the highest FloodSense tier in the hour before the flood"),
    "floodsense_peak_chance": ("DOUBLE", "the highest FloodSense chance in the hour before the flood"),
    "scored_by": (
        "STRING",
        "which FloodSense scored it: 'out of fold' (a version trained without this storm) or the live version",
    ),
    "rule_lead_min": ("DOUBLE", "the same lead for the rainfall rule; null if it had no warning in force"),
    "rule_before_pub": ("BOOLEAN", "whether the rainfall rule's warning started before PUB's first alert"),
}
EVENTS_COLUMNS = {
    **IDENTITY,
    "event_kind": (
        "STRING",
        "'pub_alert': an event of PUB alerts of any kind, timed at its first; 'flash_flood': timed at its first "
        "flash flood",
    ),
    "area": ("STRING", AREA_DOC),
    "event_time": ("TIMESTAMP_NTZ", "PUB's first alert of the event, or its first flash flood (SGT)"),
    "pub_first_alert": ("TIMESTAMP_NTZ", "PUB's first alert of the event (SGT)"),
    "pub_alerts": ("BIGINT", "PUB alerts in the event"),
    "warned_from": (
        "TIMESTAMP_NTZ",
        "start of the warning in force at the event (SGT): its first rating, each rating warning for the next hour; "
        "null if none",
    ),
    "lead_min": ("DOUBLE", "minutes from the warning's start to the event; null if no warning was in force"),
    "pub_warning_min": ("DOUBLE", "minutes from PUB's first alert to the event: 0 for a pub_alert event"),
    "warned_before_pub": ("BOOLEAN", "whether the warning started before PUB's first alert"),
    "judged": (
        "BOOLEAN",
        "whether the event counts in the scoreboard: false where PUB posts were lost or the model's inputs were "
        "missing",
    ),
}
WARNINGS_COLUMNS = {
    **IDENTITY,
    "area": ("STRING", AREA_DOC),
    "warning_start": ("TIMESTAMP_NTZ", "the first 5-minute tick at the warning level (SGT)"),
    "last_warned_tick": ("TIMESTAMP_NTZ", "the last 5-minute tick at the warning level (SGT)"),
    "in_force_until": ("TIMESTAMP_NTZ", "when the warning lapsed: an hour after its last warned tick (SGT)"),
    "warned_ticks": ("BIGINT", "5-minute ticks at the warning level"),
    "hours_in_force": ("DOUBLE", "hours from the first warned tick to when the warning lapsed"),
    "followed_by_pub_alert": ("BOOLEAN", "whether a PUB alert came in the area while the warning was in force"),
    "false_alarm": ("BOOLEAN", "a judged warning that no PUB alert followed"),
    "judged": (
        "BOOLEAN",
        "whether the warning counts: false when PUB posts during it were lost and no alert confirmed it",
    ),
}
TABLES = {
    "scoreboard": (
        SCOREBOARD_COLUMNS,
        "FloodSense's scoreboard against PUB's alerts: one row per period, model and warning level (ml/publish.py)",
    ),
    "scoreboard_floods": (
        FLOODS_COLUMNS,
        "Every flash flood PUB reported, with PUB's own warning, FloodSense's lead and the rainfall rule's "
        "(ml/publish.py)",
    ),
    "scoreboard_events": (
        EVENTS_COLUMNS,
        "Every PUB alert event and flash flood, and whether a warning came before PUB's first alert, per period, "
        "model and warning level (ml/publish.py)",
    ),
    "scoreboard_warnings": (
        WARNINGS_COLUMNS,
        "Every warning after training, per model and warning level, and whether a PUB alert followed it "
        "(ml/publish.py)",
    ),
}


# --- the tables, from scoreboards and the training run ----------------------------------------------------------------


def summary_row(metrics: dict, prefix: str = "") -> dict:
    """The scoreboard table's numbers from Scoreboard.metrics(), or from a training run's metrics logged under
    `prefix` (oof_, oof_high_, rule_dev_). Anything not logged is NaN. Shares are held to [0, 1]: a Wilson interval
    at 0 of n comes out a rounding error below zero, which would show as -0%."""
    row = {column: _number(metrics.get(prefix + key)) for key, column in SUMMARY.items()}
    for column in SHARES:
        row[column] = float(np.clip(row[column], 0.0, 1.0))  # NaN stays NaN
    return row


def followed_share(board) -> float:
    """The share of a scoreboard's judged Medium warnings that a PUB alert followed (train._followed_share)."""
    medium = board.episodes[(board.episodes["tier"] == "Medium") & board.episodes["judged"]]
    return float(medium["hit"].mean()) if len(medium) else math.nan


def development_summary(
    metrics: dict, model_name: str, rule_text: str, start: date, end: date
) -> pd.DataFrame:
    """The development rows of the scoreboard table, from the training run's out-of-fold metrics."""
    rows = []
    for model, level, prefix in DEVELOPMENT_ROWS:
        rows.append(
            {
                "period": DEVELOPMENT,
                "model": model,
                "model_version": model_name if model == FLOODSENSE else rule_text,
                "warning_level": level,
                "period_start": start,
                "period_end": end,
                **summary_row(metrics, prefix),
                "warnings_followed_by_alert": _number(metrics.get(FOLLOWED.get(prefix, ""))),
            }
        )
    return pd.DataFrame(rows)


def flood_rows(
    model: pd.DataFrame, rule: pd.DataFrame | None, period: str, scored_by: str
) -> pd.DataFrame:
    """scoreboard_floods rows from FloodSense's floods.flood_table (or oof_floods.json read back) and the rule's, joined
    on the flood; the rule's columns are null without one."""
    out = pd.DataFrame(
        {
            "flood_time": _naive(model["flood_ts"]),
            "area": model["area"].astype(object),
            "period": period,
            "judged": model["judged"].astype(bool),
            "pub_first_alert": _naive(model["first_alert"]),
            "pub_warning_min": pd.to_numeric(model["pub_lead_min"]),
            "floodsense_warned_from": _naive(model["warned_from"]),
            "floodsense_lead_min": pd.to_numeric(model["lead_min"]),
            "floodsense_before_pub": model["before_pub"].astype(bool),
            "public_warning_min": pd.to_numeric(model["public_warning_min"]),
            "floodsense_tier_at_flood": model["tier_at_flood"].map(SHOWN_TIER),
            "floodsense_chance_at_flood": pd.to_numeric(model["p_alert_at_flood"]),
            "floodsense_peak_tier": model["peak_tier"].map(SHOWN_TIER),
            "floodsense_peak_chance": pd.to_numeric(model["peak_p_alert"]),
            "scored_by": scored_by,
        }
    )
    if rule is None:
        return out.assign(rule_lead_min=np.nan, rule_before_pub=None)
    theirs = pd.DataFrame(
        {
            "flood_time": _naive(rule["flood_ts"]),
            "area": rule["area"].astype(object),
            "rule_lead_min": pd.to_numeric(rule["lead_min"]),
            "rule_before_pub": rule["before_pub"].astype(bool),
        }
    )
    return out.merge(theirs, on=["flood_time", "area"], how="left")


def event_rows(events: pd.DataFrame) -> pd.DataFrame:
    """scoreboard_events rows (without the identity columns) from Scoreboard.events, or oof_events.json read back."""
    return pd.DataFrame(
        {
            "event_kind": events["kind"].map(EVENT_KIND),
            "area": events["area"].astype(object),
            "event_time": _naive(events["time"]),
            "pub_first_alert": _naive(events["first_alert"]),
            "pub_alerts": pd.to_numeric(events["n_alerts"]).astype("Int64"),
            "warned_from": _naive(events["warned_from"]),
            "lead_min": pd.to_numeric(events["lead_min"]),
            "pub_warning_min": pd.to_numeric(events["pub_lead_min"]),
            "warned_before_pub": events["before_pub"].astype(bool),
            "judged": events["judged"].astype(bool),
        }
    )


def warning_rows(episodes: pd.DataFrame) -> pd.DataFrame:
    """scoreboard_warnings rows (without the identity columns) from a scoreboard's Medium episodes: the warnings a
    PUB alert of any kind confirms (Scoreboard.episodes)."""
    medium = episodes[episodes["tier"] == "Medium"]
    hit, judged = medium["hit"].astype(bool), medium["judged"].astype(bool)
    return pd.DataFrame(
        {
            "area": medium["area"].astype(object),
            "warning_start": _naive(medium["start"]),
            "last_warned_tick": _naive(medium["last"]),
            "in_force_until": _naive(medium["until"]),
            "warned_ticks": medium["ticks"].astype("Int64"),
            "hours_in_force": (medium["until"] - medium["start"]) / pd.Timedelta(hours=1),
            "followed_by_pub_alert": hit,
            "false_alarm": judged & ~hit,
            "judged": judged,
        }
    ).reset_index(drop=True)


def development_tables(
    metrics: dict,
    oof_events: pd.DataFrame,
    oof_floods: pd.DataFrame,
    verify_floods: pd.DataFrame | None,
    model_name: str,
    rule_text: str,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """The development rows of the scoreboard, floods and events tables, from the training run (and the rule's lead
    before each flood from the verify run's floods.json, its *_rule columns, when there is one)."""
    rule = None
    if verify_floods is not None and "lead_min_rule" in verify_floods:
        rule = verify_floods[["area", "flood_ts", "lead_min_rule", "before_pub_rule"]].rename(
            columns={"lead_min_rule": "lead_min", "before_pub_rule": "before_pub"}
        )
    identity = {"period": DEVELOPMENT, "model": FLOODSENSE, "model_version": model_name, "warning_level": MODERATE}
    return {
        "scoreboard": development_summary(metrics, model_name, rule_text, start, end),
        "scoreboard_floods": flood_rows(oof_floods, rule, DEVELOPMENT, OUT_OF_FOLD),
        "scoreboard_events": event_rows(oof_events).assign(**identity),
    }


def after_training_tables(
    model_preds: pd.DataFrame,
    rule_preds: pd.DataFrame,
    alerts: pd.DataFrame,
    window: dict,
    model_name: str,
    rule_text: str,
) -> dict[str, pd.DataFrame]:
    """The after-training rows of all four tables. `model_preds` and `rule_preds` hold every warned tick in the window
    plus the ticks in the hour before each flash flood (`kept`), whose chance flood_table reads; `window` holds start,
    end, missing and blind for scoreboard.score."""
    boards = {
        (FLOODSENSE, MODERATE): score(_warned(model_preds), alerts, **window),
        (FLOODSENSE, ELEVATED): score(_high_only(model_preds), alerts, **window),
        (RULE, HEAVY_RAIN): score(_warned(rule_preds), alerts, **window),
    }
    first, last = pd.Timestamp(window["start"]).date(), (pd.Timestamp(window["end"]) - TICK).date()
    summary, events, warnings = [], [], []
    for (model, level), board in boards.items():
        identity = {
            "period": AFTER_TRAINING,
            "model": model,
            "model_version": model_name if model == FLOODSENSE else rule_text,
            "warning_level": level,
        }
        summary.append(
            {
                **identity,
                "period_start": first,
                "period_end": last,
                **summary_row(board.metrics()),
                "warnings_followed_by_alert": followed_share(board),
            }
        )
        events.append(event_rows(board.events).assign(**identity))
        warnings.append(warning_rows(board.episodes).assign(**identity))
    return {
        "scoreboard": pd.DataFrame(summary),
        "scoreboard_floods": flood_rows(
            floods.flood_table(model_preds, alerts, **window),
            floods.flood_table(rule_preds, alerts, **window),
            AFTER_TRAINING,
            model_name,
        ),
        "scoreboard_events": pd.concat(events, ignore_index=True),
        "scoreboard_warnings": pd.concat(warnings, ignore_index=True),
    }


# --- scoring after training --------------------------------------------------------------------------------------------


def flood_hours(alerts: pd.DataFrame, start, end) -> pd.MultiIndex:
    """(area, tick) of the hour before every flash flood in [start, end): the ticks whose chance flood_table shows."""
    ev = alert_events(alerts)
    t = ev["first_flood"]
    ev = ev[t.notna() & (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))]
    keys = [(area, tick) for area, flood in zip(ev["area"], ev["first_flood"]) for tick in floods.hour_before(flood)]
    return pd.MultiIndex.from_tuples(keys, names=["area", "ts"]) if keys else _no_keys()


def kept(preds: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    """The prediction rows publishing needs: every warned tick, and the ticks in `keys` (flood_hours) whatever their
    tier."""
    at = pd.MultiIndex.from_arrays([preds["area"].astype(object), pd.to_datetime(preds["ts"])])
    keep = (preds["tier"] != "Low").to_numpy() | at.isin(keys)
    return preds.loc[keep, ["area", "ts", "tier", "p_alert"]]


def month_spans(start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[first, last) spans covering [start, end), one per calendar month, so a month of area-ticks is scored at once."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cuts = [start, *(m for m in pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS") if start < m < end), end]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if a < b]


def _warned(preds: pd.DataFrame) -> pd.DataFrame:
    return preds[preds["tier"] != "Low"]


def _high_only(preds: pd.DataFrame) -> pd.DataFrame:
    """High ticks as the only warnings, so the scoreboard judges Elevated as it judges Moderate (train._high_only)."""
    return preds[preds["tier"] == "High"].assign(tier="Medium")


def _no_keys() -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["area", "ts"])


def _naive(ts: pd.Series) -> pd.Series:
    """Datetimes as naive SGT at nanoseconds, NaT for missing; text from an MLflow table ('NaT' too) reads back."""
    ts = pd.to_datetime(ts.replace({"NaT": None}) if ts.dtype == object else ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    return ts.astype("datetime64[ns]")


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


# --- on Databricks -------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import mlflow
    from pyspark.sql import SparkSession

    from .live import newest, rule_from_name
    from .models import load_logged
    from .train import _missing_ticks
    from .verify import _artifact_table

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")  # naive pandas times are SGT
    gold, silver, ml = (f"{args.catalog}.{s}" for s in (args.gold_schema, args.silver_schema, args.ml_schema))
    full = f"{ml}.{args.model}"
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(full, args.alias)
    name = C.version_name(args.model, mv.version, mv.aliases)
    training = client.get_run(mv.run_id).data
    rule = rule_from_name(training.params["rule"])
    rule_text = f"{rule.window_min}-minute rain of {rule.medium_mm:g} mm or more within {C.GAUGE_RADIUS_KM:g} km"
    dev_start, dev_end = (date.fromisoformat(d) for d in training.params["development"].split(" to "))
    verify_run = _verify_run(mlflow, args.experiment, full, mv.version)
    log.info("publishing %s v%s (@%s), run %s, verified in %s", full, mv.version, args.alias, mv.run_id, verify_run)

    tables = development_tables(
        training.metrics,
        _artifact_table(mlflow, mv.run_id, "oof_events.json"),
        _artifact_table(mlflow, mv.run_id, "oof_floods.json"),
        _artifact_table(mlflow, verify_run, "floods.json") if verify_run else None,
        name,
        rule_text,
        dev_start,
        dev_end,
    )

    # after training: to the newest tick both gold's labels and its features are known for
    start = pd.Timestamp(C.TEST_START)
    end = min(newest(spark, f"{gold}.{C.LABEL_TICKS}", "ts"), newest(spark, f"{gold}.{C.FEATURE_TICKS}", "ts")) + TICK
    alerts = spark.read.table(f"{silver}.{C.AREA_ALERTS}").select("area", "alert_ts", "kind").toPandas()
    window = {
        "start": start,
        "end": end,
        "missing": _missing_ticks(spark, f"{gold}.{C.LABEL_TICKS}", "label_window"),
        "blind": _missing_ticks(spark, f"{gold}.{C.FEATURE_TICKS}", "feature_window"),
    }
    root = mlflow.artifacts.download_artifacts(f"models:/{full}/{mv.version}", dst_path=tempfile.mkdtemp())
    model = load_logged(Path(root))
    model_preds, rule_preds = score_after_training(spark, gold, model, rule, alerts, start, end)
    after = after_training_tables(model_preds, rule_preds, alerts, window, name, rule_text)

    for table, (columns, comment) in TABLES.items():
        frame = pd.concat([t[table] for t in (tables, after) if table in t], ignore_index=True)
        write_table(spark, f"{ml}.{table}", frame, columns, comment)
        log.info("wrote %s.%s: %d rows", ml, table, len(frame))
    create_views(spark, ml, silver, gold)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    shown = ["period", "model", "warning_level", "pub_alert_events", "alerts_warned_before_pub", "alert_mean_lead_min"]
    shown += ["flash_floods", "floods_warned_before_pub", "pub_flood_warning_min", "false_alarms_per_area_month"]
    board = pd.concat([tables["scoreboard"], after["scoreboard"]], ignore_index=True)
    log.info("scoreboard from %s to %s:\n%s", start, end, board[shown].round(3).to_string())
    return 0


def score_after_training(spark, gold: str, model, rule, alerts: pd.DataFrame, start, end):
    """(FloodSense's, the rule's) kept prediction rows (`kept`) over [start, end), scored a month at a time from
    gold.area_features, as verify.py scores rows: what live scoring would have said at each tick."""
    from pyspark.sql import functions as F

    from .verify import _as_rows

    keys = flood_hours(alerts, start, end)
    features = spark.read.table(f"{gold}.{C.AREA_FEATURES}").select("area", "ts", *C.MODEL_FEATURES)
    ours, theirs = [], []
    for first, last in month_spans(start, end):
        # literals as text, which Spark reads in the session time zone (SGT)
        lo, hi = (F.lit(t.strftime("%Y-%m-%d %H:%M:%S")).cast("timestamp") for t in (first, last))
        rows = _as_rows(features.where((F.col("ts") >= lo) & (F.col("ts") < hi)).toPandas())
        ours.append(kept(model.predict(rows), keys))
        theirs.append(kept(rule.predict(rows), keys))
        log.info("scored %s to %s: %d rows, %d warned", first, last, len(rows), int((ours[-1]["tier"] != "Low").sum()))
    return pd.concat(ours, ignore_index=True), pd.concat(theirs, ignore_index=True)


def write_table(spark, table: str, frame: pd.DataFrame, columns: dict, comment: str) -> None:
    """Replace `table` with `frame`, each column typed and described as `columns` says ({name: (SQL type, comment)})."""
    frame = frame[list(columns)].copy()
    for column, (kind, _) in columns.items():
        values = frame[column]
        if kind == "BIGINT":  # counts logged to MLflow come back as floats
            values = pd.to_numeric(values).round().astype("Int64")
        if kind != "TIMESTAMP_NTZ":  # NaT already arrives as NULL
            # Spark keeps a pandas NaN as NaN, and a missing count or flag should be NULL
            values = values.astype(object).where(values.notna(), None)
        frame[column] = values
    staged = spark.createDataFrame(frame, schema=", ".join(f"{c} {kind}" for c, (kind, _) in columns.items()))
    view = f"publish_{table.rsplit('.', 1)[-1]}"
    staged.createOrReplaceTempView(view)
    spec = ", ".join(f"{c} {kind} COMMENT {_quoted(doc)}" for c, (kind, doc) in columns.items())
    spark.sql(f"CREATE OR REPLACE TABLE {table} ({spec}) COMMENT {_quoted(comment)}")
    spark.sql(f"INSERT INTO {table} SELECT * FROM {view}")


def create_views(spark, ml: str, silver: str, gold: str) -> None:
    """pub_alerts, flood_prone_trend and live_risk beside the scoreboard tables, their times SGT in any session."""

    def sgt(column: str) -> str:
        # A view keeps the session settings it was created in (this job's are SGT), while current_timezone() in it
        # is the reader's: mixing the two shifted live_risk by 16 hours. Epoch seconds and Singapore's fixed +08:00
        # (since 1982) need no time zone at all.
        return f"TIMESTAMP_NTZ'1970-01-01 08:00:00' + make_dt_interval(0, 0, 0, unix_seconds({column}))"

    def described(columns: dict[str, str]) -> str:
        return ", ".join(f"{c} COMMENT {_quoted(doc)}" for c, doc in columns.items())

    for view, columns, comment, query in (
        (
            "pub_alerts",
            {
                "alert_time": "when PUB issued the alert (SGT)",
                "area": AREA_DOC,
                "kind": "warning: a drain or canal at 90% of capacity (Risk of Flash Floods); flash_flood: a flash "
                "flood occurred",
                "alert_type": "the kind in words: Risk of flash floods (drain at 90%), or Flash flood",
                "location": "the place PUB named",
                "lat": "latitude (WGS84) of the place PUB named",
                "lon": "longitude (WGS84) of the place PUB named",
                "source": "telegram (the PUB Flood Alerts channel) or api (data.gov.sg)",
            },
            "Every PUB flood alert since Nov 2021, placed in an NEA forecast area (silver.area_alerts)",
            f"""SELECT {sgt("alert_ts")}, area, kind,
                       CASE kind WHEN 'flash_flood' THEN 'Flash flood' ELSE 'Risk of flash floods (drain at 90%)' END,
                       location, lat, lon, source
                FROM {silver}.{C.AREA_ALERTS}""",
        ),
        (
            "flood_prone_trend",
            {
                "year": "year of PUB's list",
                "hectares": "land PUB designated flood-prone that year, in hectares",
                "change_ha": "change from the year before, in hectares; null in the first year, which has none before it",
                "change_pct": "change from the year before, in percent; null in the first year",
            },
            "Flood-prone land in Singapore by year, from PUB (data.gov.sg; gold.flood_prone_trend)",
            f"SELECT year, hectares, change_ha, change_pct FROM {gold}.flood_prone_trend",
        ),
        (
            "live_risk",
            {
                "area": AREA_DOC,
                "forecast_time": "the newest 5-minute tick FloodSense rated (SGT)",
                "risk": "the warning in force, as the app's map shows it: Elevated for an hour after any Elevated "
                "rating, else Moderate for an hour after any Moderate one, else Low",
                "latest_rating": "the newest 5-minute rating alone: Low, Moderate or Elevated",
                "warning_until": "when the warning in force lapses unless FloodSense rates the area again (SGT); null "
                "at Low",
                "chance": "FloodSense's newest chance (0 to 1) of a PUB flood alert in the area within the hour",
                "reason": "the plain-English reason the app shows for the warning in force",
                "model": "the FloodSense version that rated it",
            },
            "Every area's FloodSense warning now, as the app shows it: each Moderate or Elevated rating is a warning "
            "for the next hour, and Elevated has priority (ml.predictions; ml/broadcast.py)",
            f"""WITH newest AS (
                  SELECT model, ts FROM {ml}.{C.PREDICTIONS} WHERE model NOT LIKE 'rainfall_rule%'
                  ORDER BY ts DESC LIMIT 1),
                hour AS (
                  SELECT p.area, p.ts, p.tier, p.p_alert, p.reason, p.model
                  FROM {ml}.{C.PREDICTIONS} p JOIN newest n ON p.model = n.model
                  WHERE p.ts > n.ts - INTERVAL 60 MINUTES AND p.ts <= n.ts),
                per_area AS (
                  SELECT area, max(ts) AS ts, max_by(tier, ts) AS rated, max_by(p_alert, ts) AS chance,
                         max(CASE WHEN tier = 'High' THEN ts END) AS high_ts,
                         max(CASE WHEN tier <> 'Low' THEN ts END) AS warned_ts,
                         max_by(reason, CASE WHEN tier = 'High' THEN ts END) AS high_reason,
                         max_by(reason, CASE WHEN tier <> 'Low' THEN ts END) AS warned_reason,
                         max(model) AS model
                  FROM hour GROUP BY area)
                SELECT area, {sgt("ts")},
                       CASE WHEN high_ts IS NOT NULL THEN 'Elevated' WHEN warned_ts IS NOT NULL THEN 'Moderate'
                            ELSE 'Low' END,
                       CASE rated WHEN 'Medium' THEN 'Moderate' WHEN 'High' THEN 'Elevated' ELSE rated END,
                       {sgt("coalesce(high_ts, warned_ts) + INTERVAL 60 MINUTES")},
                       chance,
                       CASE WHEN high_ts IS NOT NULL THEN high_reason ELSE warned_reason END,
                       model
                FROM per_area""",
        ),
    ):
        spark.sql(
            f"CREATE OR REPLACE VIEW {ml}.{view} ({described(columns)}) COMMENT {_quoted(comment)} AS {query}"
        )


def _verify_run(mlflow, experiment: str, full: str, version) -> str | None:
    """The newest flood-verify run of this version, which holds floods.json, or None."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"params.model = '{full}' AND params.version = '{version}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return None if runs.empty else runs.iloc[0]["run_id"]


def _quoted(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ml-schema", default="ml")
    parser.add_argument("--model", default=C.REGISTERED_MODEL, help="registered model name in <catalog>.<ml schema>")
    parser.add_argument("--experiment", required=True, help="MLflow experiment the training and verify runs are in")
    parser.add_argument("--alias", default=C.CHAMPION, help="the version to publish (default %(default)s)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
