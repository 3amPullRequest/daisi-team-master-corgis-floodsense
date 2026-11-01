"""Publish the live model's scoreboard as tables, for the agencies' AI/BI dashboard, its Genie space and the app.

The flood-scoreboard job runs it whenever @champion moves (resources/scoreboard.job.yml):

    databricks bundle run scoreboard -t dev -p flood_dev

It scores the test period: from contract.TEST_START (20 Dec 2025), after the training data ended, to the newest tick
gold has labels and features for. Every area-tick of gold.area_features is scored by the champion and by the rule its
training run fitted, as live scoring would have scored it; nothing is cross-validated on the training data. (Its 2026
floods were looked at while versions were chosen, verify's floods.json, so it is the record after training rather than
an untouched test.)

Every Moderate or Elevated rating is a warning for the next hour, so a warning runs from its first rating to an hour
after its last (scoreboard.py). The app's storm replays, its live map and its Telegram posts count warnings the same way
(broadcast.py). The measures are about flash floods, each PUB report counted on its own (floods.report_table), and
about the warnings themselves:

- floods caught: a warning was in force when the flood struck, and its first rating came 5 minutes to an hour before;
- the lead time: from that first rating to the flood, averaged over the floods a warning was in force at;
- the chance of a PUB alert given a warning, and of a flash flood: the share of warnings that one came within.

It replaces three tables in the ml schema, their columns described for Genie (the *_COLUMNS below):

- scoreboard: a row per model and warning level (FloodSense from Moderate, FloodSense at Elevated, the rule), with
  those measures and their 95% intervals;
- scoreboard_floods: a row per flash flood PUB reported, with FloodSense's lead and the rule's;
- scoreboard_warnings: a row per warning, per model and level, and whether a PUB alert or a flash flood followed it;

drops the tables it no longer publishes (DROPPED), and puts three views beside them, so the dashboard and Genie read
one schema: pub_alerts (silver.area_alerts), flood_prone_trend (gold) and live_risk (the newest tick of
ml.predictions). Times are SGT wall-clock times (TIMESTAMP_NTZ, whatever the reader's session time zone), and tiers are
named as the app shows them: Low, Moderate (Medium in the data) and Elevated (High).
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract as C
from . import floods
from .scoreboard import TICK, _bounds, _confirmed, _ns, score, wilson

log = logging.getLogger(__name__)

AFTER_TRAINING = "after_training"
FLOODSENSE, RULE = "FloodSense", "Rainfall rule"
# what each row counts as a warning
MODERATE, ELEVATED, HEAVY_RAIN = "Moderate or above", "Elevated", "Heavy rain"
SHOWN_TIER = {"Low": "Low", "Medium": "Moderate", "High": "Elevated"}  # the app's names (frontend lib/content.ts)
# Published until 1 Nov 2026 and dropped since, so no reader finds them stale: scoreboard_events held PUB's alert
# events and the lead before PUB's first alert, which the scoreboard no longer measures.
DROPPED = ("scoreboard_events",)

PERIOD_DOC = "'after_training': from 20 Dec 2025, after the training data ended: storms the live model never trained on"
AREA_DOC = "NEA 2-hour forecast area (one of 47), where PUB's alert location falls"
IDENTITY = {
    "period": ("STRING", PERIOD_DOC),
    "model": ("STRING", "'FloodSense', or 'Rainfall rule': a simple rule that warns once heavy rain has fallen nearby"),
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
    "flash_floods": (
        "BIGINT",
        "flash floods PUB reported, each report on its own (two roads flooding minutes apart are two), where the "
        "model's inputs were known for the hour before",
    ),
    "floods_warned": ("BIGINT", "of those, the floods a warning was in force at when they struck"),
    "floods_caught": (
        "BIGINT",
        "of those, the floods caught: the warning's first rating came 5 minutes to an hour before the flood (less came "
        "as it happened; more, after the hour that rating warned of)",
    ),
    "floods_caught_share": ("DOUBLE", "floods_caught over flash_floods (0 to 1)"),
    "floods_caught_share_lo": ("DOUBLE", "95% interval for floods_caught_share, low end (Wilson)"),
    "floods_caught_share_hi": ("DOUBLE", "95% interval for floods_caught_share, high end (Wilson)"),
    "flood_mean_lead_min": (
        "DOUBLE",
        "mean lead time: minutes from the first rating of the warning in force to the flood, over the floods a warning "
        "was in force at (each Moderate or Elevated rating is a warning for the next hour, so a warning starts at its "
        "first rating)",
    ),
    "warnings": (
        "BIGINT",
        "warnings judged: warned 5-minute ticks in one area at most an hour apart are one warning",
    ),
    "warnings_followed_by_alert": (
        "DOUBLE",
        "the chance of a PUB alert given a warning: the share of warnings (0 to 1) that a PUB alert of any kind in the "
        "area came within; the rest are false alarms",
    ),
    "warnings_followed_by_alert_lo": ("DOUBLE", "95% interval for warnings_followed_by_alert, low end (Wilson)"),
    "warnings_followed_by_alert_hi": ("DOUBLE", "95% interval for warnings_followed_by_alert, high end (Wilson)"),
    "warnings_followed_by_flood": (
        "DOUBLE",
        "the chance of a flash flood given a warning: the share of warnings (0 to 1) that a flash flood in the area "
        "came within",
    ),
    "warnings_followed_by_flood_lo": ("DOUBLE", "95% interval for warnings_followed_by_flood, low end (Wilson)"),
    "warnings_followed_by_flood_hi": ("DOUBLE", "95% interval for warnings_followed_by_flood, high end (Wilson)"),
}
SHARES = tuple(c for c in SCOREBOARD_COLUMNS if c.startswith(("floods_caught_share", "warnings_followed_by")))
FLOODS_COLUMNS = {
    "flood_time": ("TIMESTAMP_NTZ", "when PUB reported the flash flood (SGT)"),
    "area": ("STRING", AREA_DOC),
    "location": ("STRING", "the place PUB named"),
    "period": ("STRING", PERIOD_DOC),
    "judged": (
        "BOOLEAN",
        "whether the flood counts in the scoreboard: false where the model's inputs were missing in the hour before it",
    ),
    "floodsense_warned_from": (
        "TIMESTAMP_NTZ",
        "first rating of the FloodSense warning (Moderate or above) in force at the flood (SGT); null if none",
    ),
    "floodsense_lead_min": (
        "DOUBLE",
        "lead time: minutes from that first rating to the flood; null if no FloodSense warning was in force",
    ),
    "floodsense_caught": (
        "BOOLEAN",
        "whether FloodSense caught the flood: its warning's first rating came 5 minutes to an hour before it",
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
    "scored_by": ("STRING", "the FloodSense version that scored it"),
    "rule_lead_min": ("DOUBLE", "the same lead time for the rainfall rule; null if it had no warning in force"),
    "rule_caught": ("BOOLEAN", "whether the rainfall rule caught the flood, by the same 5 minutes to an hour"),
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
    "followed_by_flash_flood": ("BOOLEAN", "whether a flash flood came in the area while the warning was in force"),
    "false_alarm": ("BOOLEAN", "a judged warning that no PUB alert followed"),
    "judged": (
        "BOOLEAN",
        "whether the warning counts: false when PUB posts during it were lost and no alert confirmed it",
    ),
}
TABLES = {
    "scoreboard": (
        SCOREBOARD_COLUMNS,
        "FloodSense's scoreboard since 20 Dec 2025: flash floods caught, lead time, and the chance of a PUB alert or a "
        "flash flood given a warning, per model and warning level (ml/publish.py)",
    ),
    "scoreboard_floods": (
        FLOODS_COLUMNS,
        "Every flash flood PUB reported since 20 Dec 2025, with FloodSense's lead and the rainfall rule's "
        "(ml/publish.py)",
    ),
    "scoreboard_warnings": (
        WARNINGS_COLUMNS,
        "Every warning since 20 Dec 2025, per model and warning level, and whether a PUB alert or a flash flood "
        "followed it (ml/publish.py)",
    ),
}


# --- the tables, from scoreboards ---------------------------------------------------------------------------------------


def warning_rows(episodes: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """scoreboard_warnings rows (without the identity columns) from a scoreboard's Medium episodes, the warnings a
    PUB alert of any kind confirms (Scoreboard.episodes), with whether a flash flood came while each was in force."""
    medium = episodes[episodes["tier"] == "Medium"]
    hit, judged = medium["hit"].astype(bool), medium["judged"].astype(bool)
    a = pd.DataFrame({"area": alerts["area"], "ts": _ns(alerts["alert_ts"]), "kind": alerts["kind"]})
    flood = a[a["kind"] == C.FLASH_FLOOD].dropna(subset=["ts"])
    return pd.DataFrame(
        {
            "area": medium["area"].astype(object),
            "warning_start": _naive(medium["start"]),
            "last_warned_tick": _naive(medium["last"]),
            "in_force_until": _naive(medium["until"]),
            "warned_ticks": medium["ticks"].astype("Int64"),
            "hours_in_force": (medium["until"] - medium["start"]) / pd.Timedelta(hours=1),
            "followed_by_pub_alert": hit,
            "followed_by_flash_flood": _confirmed(medium, flood["area"], flood["ts"]),
            "false_alarm": judged & ~hit,
            "judged": judged,
        }
    ).reset_index(drop=True)


def summary_row(warnings: pd.DataFrame, reports: pd.DataFrame) -> dict:
    """The scoreboard table's measures for one model and warning level: the floods it caught and its lead time, from
    its floods.report_table (floods.report_summary), and the chance of a PUB alert, and of a flash flood, given one of
    its judged warnings (its warning_rows), each share with a 95% Wilson interval. Shares are held to [0, 1]: a Wilson
    interval at 0 of n comes out a rounding error below zero, which would show as -0%."""
    judged = warnings[warnings["judged"].astype(bool)]
    row = {**floods.report_summary(reports), "warnings": float(len(judged))}
    for column, followed in (
        ("warnings_followed_by_alert", "followed_by_pub_alert"),
        ("warnings_followed_by_flood", "followed_by_flash_flood"),
    ):
        k = int(judged[followed].astype(bool).sum())
        row[column] = k / len(judged) if len(judged) else math.nan
        row.update(_bounds(column, wilson(k, len(judged))))
    for column in SHARES:
        row[column] = float(np.clip(row[column], 0.0, 1.0))  # NaN stays NaN
    return row


def flood_rows(model: pd.DataFrame, rule: pd.DataFrame, scored_by: str) -> pd.DataFrame:
    """scoreboard_floods rows from FloodSense's floods.report_table and the rule's, both over the same reports, so
    their rows line up one to one."""
    if list(zip(model["area"], model["flood_ts"])) != list(zip(rule["area"], rule["flood_ts"])):
        raise ValueError("FloodSense's floods and the rule's differ: score both over the same alerts and window")
    return pd.DataFrame(
        {
            "flood_time": _naive(model["flood_ts"]),
            "area": model["area"].astype(object),
            "location": model["location"],
            "period": AFTER_TRAINING,
            "judged": model["judged"].astype(bool),
            "floodsense_warned_from": _naive(model["warned_from"]),
            "floodsense_lead_min": pd.to_numeric(model["lead_min"]),
            "floodsense_caught": model["caught"].astype(bool),
            "floodsense_tier_at_flood": model["tier_at_flood"].map(SHOWN_TIER),
            "floodsense_chance_at_flood": pd.to_numeric(model["p_alert_at_flood"]),
            "floodsense_peak_tier": model["peak_tier"].map(SHOWN_TIER),
            "floodsense_peak_chance": pd.to_numeric(model["peak_p_alert"]),
            "scored_by": scored_by,
            "rule_lead_min": pd.to_numeric(rule["lead_min"]).to_numpy(),
            "rule_caught": rule["caught"].astype(bool).to_numpy(),
        }
    )


def after_training_tables(
    model_preds: pd.DataFrame,
    rule_preds: pd.DataFrame,
    alerts: pd.DataFrame,
    window: dict,
    model_name: str,
    rule_text: str,
) -> dict[str, pd.DataFrame]:
    """The three tables over the test period. `model_preds` and `rule_preds` hold every warned tick in the window plus
    the ticks in the hour before each flash flood (`kept`), whose chance report_table reads; `window` holds start, end,
    missing and blind for scoreboard.score."""
    span = {"start": window["start"], "end": window["end"], "blind": window["blind"]}
    levels = {
        (FLOODSENSE, MODERATE): _warned(model_preds),
        (FLOODSENSE, ELEVATED): _high_only(model_preds),
        (RULE, HEAVY_RAIN): _warned(rule_preds),
    }
    first, last = pd.Timestamp(window["start"]).date(), (pd.Timestamp(window["end"]) - TICK).date()
    summary, warnings = [], []
    for (model, level), preds in levels.items():
        identity = {
            "period": AFTER_TRAINING,
            "model": model,
            "model_version": model_name if model == FLOODSENSE else rule_text,
            "warning_level": level,
        }
        rows = warning_rows(score(preds, alerts, **window).episodes, alerts)
        reports = floods.report_table(preds, alerts, **span)
        summary.append(
            {**identity, "period_start": first, "period_end": last, **summary_row(rows, reports)}
        )
        warnings.append(rows.assign(**identity))
    return {
        "scoreboard": pd.DataFrame(summary),
        "scoreboard_floods": flood_rows(
            floods.report_table(model_preds, alerts, **span),
            floods.report_table(rule_preds, alerts, **span),
            model_name,
        ),
        "scoreboard_warnings": pd.concat(warnings, ignore_index=True),
    }


# --- scoring the test period -------------------------------------------------------------------------------------------


def flood_hours(alerts: pd.DataFrame, start, end) -> pd.MultiIndex:
    """(area, tick) of the hour before every flash flood PUB reported in [start, end): the ticks whose chance
    report_table shows."""
    a = alerts[alerts["kind"] == C.FLASH_FLOOD]
    t = _naive(a["alert_ts"])
    picked = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))
    keys = {(area, tick) for area, flood in zip(a["area"][picked], t[picked]) for tick in floods.hour_before(flood)}
    return pd.MultiIndex.from_tuples(sorted(keys), names=["area", "ts"]) if keys else _no_keys()


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


# --- on Databricks -------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import mlflow
    from pyspark.sql import SparkSession

    from .live import newest, rule_from_name
    from .models import load_logged
    from .train import _missing_ticks

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")  # naive pandas times are SGT
    gold, silver, ml = (f"{args.catalog}.{s}" for s in (args.gold_schema, args.silver_schema, args.ml_schema))
    full = f"{ml}.{args.model}"
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(full, args.alias)
    name = C.version_name(args.model, mv.version, mv.aliases)
    rule = rule_from_name(client.get_run(mv.run_id).data.params["rule"])
    rule_text = f"{rule.window_min}-minute rain of {rule.medium_mm:g} mm or more within {C.GAUGE_RADIUS_KM:g} km"
    log.info("publishing %s v%s (@%s), trained in run %s", full, mv.version, args.alias, mv.run_id)

    # the test period: to the newest tick both gold's labels and its features are known for
    start = pd.Timestamp(C.TEST_START)
    end = min(newest(spark, f"{gold}.{C.LABEL_TICKS}", "ts"), newest(spark, f"{gold}.{C.FEATURE_TICKS}", "ts")) + TICK
    alerts = spark.read.table(f"{silver}.{C.AREA_ALERTS}").select("area", "alert_ts", "kind", "location").toPandas()
    window = {
        "start": start,
        "end": end,
        "missing": _missing_ticks(spark, f"{gold}.{C.LABEL_TICKS}", "label_window"),
        "blind": _missing_ticks(spark, f"{gold}.{C.FEATURE_TICKS}", "feature_window"),
    }
    root = mlflow.artifacts.download_artifacts(f"models:/{full}/{mv.version}", dst_path=tempfile.mkdtemp())
    model = load_logged(Path(root))
    model_preds, rule_preds = score_after_training(spark, gold, model, rule, alerts, start, end)
    tables = after_training_tables(model_preds, rule_preds, alerts, window, name, rule_text)

    for table, (columns, comment) in TABLES.items():
        write_table(spark, f"{ml}.{table}", tables[table], columns, comment)
        log.info("wrote %s.%s: %d rows", ml, table, len(tables[table]))
    for table in DROPPED:
        spark.sql(f"DROP TABLE IF EXISTS {ml}.{table}")
    create_views(spark, ml, silver, gold)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    shown = ["model", "warning_level", "flash_floods", "floods_warned", "floods_caught", "flood_mean_lead_min"]
    shown += ["warnings", "warnings_followed_by_alert", "warnings_followed_by_flood"]
    log.info("scoreboard from %s to %s:\n%s", start, end, tables["scoreboard"][shown].round(3).to_string())
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


def _quoted(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--ml-schema", default="ml")
    parser.add_argument("--model", default=C.REGISTERED_MODEL, help="registered model name in <catalog>.<ml schema>")
    parser.add_argument("--alias", default=C.CHAMPION, help="the version to publish (default %(default)s)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
