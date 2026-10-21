"""The gated pair with and without the tide, on real data, without Databricks: a laptop stand-in for
`databricks bundle run train --params compare_without=tide` until the workspace is reachable.

It runs the production model code (gated.fit, scoreboard.score, compare.py) on rows it builds itself from the landed
files, with two simplifications against gold, so read its numbers as a rough guide and the flood-train run as the
answer:

- **One island-wide "area".** Placing alerts in NEA's 47 areas needs silver's gazetteer (OneMap, PUB sensors, hand
  fixes), so every alert and every tick here belongs to "Singapore": the label is a PUB alert anywhere in the next hour,
  rain is the wettest gauge on the island, and the forecast is the heaviest class over all 47 areas.
- **The 10-minute reading lag only.** gold also holds back a reading until NEA published it (the CSVs' update time);
  here every reading counts from 10 minutes after its timestamp.

Label windows follow gold.label_ticks (an id gap blanks the span between its neighbours, with the same margins) and
feature windows gold.feature_ticks (every reading timestamp of the hour reported, a forecast valid). The tide is
gold.tide_levels as the local Spark run built it (predicted_m, minutes_to_high_tide).

    uv run python explorations/tide_ablation_local.py --landing .landing-dev --tide-levels <warehouse>/gold.db/tide_levels
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from daisi_team_master_corgis_floodsense.labels import alert_text
from daisi_team_master_corgis_floodsense.ml import compare, gated
from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.scoreboard import score, tick_labels
from daisi_team_master_corgis_floodsense.ml.split import make_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT / "src" / "pipeline")
)  # utilities.forecast_categories, as pytest finds it

from utilities.forecast_categories import FORECASTS

log = logging.getLogger("tide_ablation")
AREA = "Singapore"
TICK = pd.Timedelta(minutes=C.TICK_MINUTES)
LAG = pd.Timedelta(minutes=C.READING_LAG_MINUTES)
SLOTS = max(C.RAIN_WINDOWS_MINUTES) // C.TICK_MINUTES
END = pd.Timestamp("2025-01-01")  # the yearly CSVs stop at 2024
SEEDS = (0, 1, 2)
# Shares of development ticks each fit warns on, highest p_alert first. One island-wide "area" makes the scoreboard's
# false-alarm budget (per area-month, over 47 areas) meaningless, and the cut-offs it picks keep a warning on for days,
# so the fits are compared at the same warning budget instead.
SHARES = (0.02, 0.05, 0.10)
PAIRED_SHARE = 0.05


def sgt_naive(text: pd.Series) -> pd.Series:
    """ISO times with an offset -> naive SGT, as contract.py reads naive times."""
    return (
        pd.to_datetime(text, utc=True, format="ISO8601")
        .dt.tz_convert("Asia/Singapore")
        .dt.tz_localize(None)
    )


# --- labels -----------------------------------------------------------------------------------------------------------


def posts(landing: str) -> pd.DataFrame:
    """One row per message_id, its latest version (a web archive copy was seen when captured), as silver keeps it."""
    lines = []
    for path in glob.glob(f"{landing}/pub_telegram_posts/**/*.jsonl*", recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            lines += [json.loads(line) for line in f if line.strip()]
    frame = pd.DataFrame(lines)
    frame["seen_at"] = pd.to_datetime(
        frame.get("captured_at", pd.Series(dtype=str)).fillna(frame["fetched_at"]),
        utc=True,
        format="ISO8601",
    )
    frame = frame.dropna(subset=["message_id"]).sort_values("seen_at")
    latest = frame.drop_duplicates("message_id", keep="last").sort_values("message_id")
    return latest.reset_index(drop=True)


def alerts_from(latest: pd.DataFrame) -> pd.DataFrame:
    """silver.area_alerts with every location in one area: one row per alert location, timed by its stamp (else the
    posting), kinds as contract.ALERT_KINDS; a hand label in data/static wins over the patterns."""
    manual = pd.read_csv(
        ROOT / "data/static/telegram_manual_labels.csv", comment="#"
    ).set_index("message_id")["category"]
    rows = []
    for post in latest.itertuples():
        service = bool(getattr(post, "service", False) or False)
        parsed = alert_text.parse_row(post.text, post.posted_at, service)
        category = manual.get(int(post.message_id), parsed["category"])
        if category not in C.ALERT_KINDS:
            continue
        for loc in parsed["locations"]:
            rows.append(
                {
                    "area": AREA,
                    "alert_ts": loc["stamp_ts"] or post.posted_at,
                    "kind": category,
                }
            )
    alerts = pd.DataFrame(rows)
    alerts["alert_ts"] = sgt_naive(alerts["alert_ts"])
    return alerts


def missing_label_ticks(latest: pd.DataFrame, ticks: pd.DatetimeIndex) -> pd.Series:
    """gold.label_ticks' rule: an id gap blanks the posting span between its neighbours, and a tick is missing when
    any posting time from STAMP_AHEAD before it to HORIZON + POST_DELAY_MAX after it falls in such a span."""
    ids = latest["message_id"].astype(int).to_numpy()
    posted = sgt_naive(latest["posted_at"]).to_numpy()
    gap = np.flatnonzero(np.diff(ids) > 1)
    before = pd.Timedelta(minutes=C.STAMP_AHEAD_MINUTES)
    after = pd.Timedelta(minutes=C.HORIZON_MINUTES + C.POST_DELAY_MAX_MINUTES)
    missing = np.zeros(len(ticks), dtype=bool)
    t = ticks.to_numpy()
    for i in gap:
        lo, hi = posted[i], posted[i + 1]
        # the tick's posting window [t - before, t + after] overlaps (lo, hi)
        missing |= (t - before < hi) & (t + after > lo)
    return pd.Series(ticks[missing])


# --- features ---------------------------------------------------------------------------------------------------------


def rain(landing: str, ticks: pd.DatetimeIndex) -> pd.DataFrame:
    """Island-wide rain at each tick: the largest 15/30/60-minute total of any gauge, from readings timestamped in the
    window ending READING_LAG before the tick, the gauges reporting in it, and how many of the window's 12 reading
    timestamps any gauge reported at (gold.feature_ticks' rain check)."""
    parts = []
    for path in sorted(glob.glob(f"{landing}/nea_rainfall_hist/year=*/*.csv")):
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                include_columns=["timestamp", "station_id", "reading_value"],
                column_types={"timestamp": pa.string(), "station_id": pa.string()},
            ),
        ).to_pandas()
        # '2021-01-01T00:00:00+08:00': SGT wall time is the first 19 characters
        table["ts"] = pd.to_datetime(
            table["timestamp"].str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S"
        )
        parts.append(table[["ts", "station_id", "reading_value"]])
        log.info("rain: %s, %d readings", Path(path).parent.name, len(table))
    readings = pd.concat(parts, ignore_index=True)
    readings = readings[
        readings["ts"].dt.second.eq(0) & readings["ts"].dt.minute.mod(5).eq(0)
    ]
    grid = pd.date_range(ticks[0] - LAG - SLOTS * TICK, ticks[-1], freq=TICK)
    mm = (
        readings.pivot_table(
            index="ts", columns="station_id", values="reading_value", aggfunc="last"
        )
        .reindex(grid)
        .astype("float64")
    )
    out = pd.DataFrame(index=grid)
    for w in C.RAIN_WINDOWS_MINUTES:
        sums = mm.rolling(w // C.TICK_MINUTES, min_periods=1).sum()
        out[C.rain_column(w)] = sums.where(
            mm.notna().rolling(w // C.TICK_MINUTES).sum() > 0
        ).max(axis=1)
    reported = mm.notna()
    out["rain_gauges"] = reported.rolling(SLOTS, min_periods=1).max().sum(axis=1)
    out["rain_stamps"] = (
        reported.any(axis=1).astype(int).rolling(SLOTS, min_periods=1).sum()
    )
    # the window ending READING_LAG before the tick
    out.index = out.index + LAG
    return out.reindex(ticks)


def forecast(landing: str, ticks: pd.DatetimeIndex) -> pd.DataFrame:
    """The heaviest rain class among the 47 areas' newest valid forecasts at each tick (published by it, its valid
    period not over), and the age of the newest publication."""
    frames = [
        pd.read_csv(
            p,
            usecols=[
                "timestamp",
                "update_timestamp",
                "valid_period_end",
                "location_name",
                "forecast_text",
            ],
            dtype=str,
        )
        for p in sorted(glob.glob(f"{landing}/nea_two_hr_forecast_hist/year=*/*.csv"))
    ]
    f = pd.concat(frames, ignore_index=True)
    f["updated"] = sgt_naive(f["update_timestamp"])
    f["valid_to"] = sgt_naive(f["valid_period_end"])
    f["issued"] = sgt_naive(f["timestamp"])
    f["rain_class"] = f["forecast_text"].map({t: r for t, (_, r) in FORECASTS.items()})
    f = f.dropna(subset=["updated", "valid_to", "rain_class"]).sort_values(
        ["updated", "issued"]
    )
    best = np.full(len(ticks), -1.0)
    newest = np.full(
        len(ticks), -1, dtype="int64"
    )  # ns of the newest valid publication; -1 for none
    left = pd.DataFrame({"ts": ticks})
    for _, area in f.groupby("location_name"):
        m = pd.merge_asof(
            left,
            area[["updated", "valid_to", "rain_class"]],
            left_on="ts",
            right_on="updated",
        )
        valid = (m["ts"] < m["valid_to"]).to_numpy()
        best = np.maximum(
            best, np.where(valid, m["rain_class"].to_numpy(dtype=float), -1.0)
        )
        updated = m["updated"].to_numpy().astype("datetime64[ns]").astype("int64")
        newest = np.maximum(newest, np.where(valid, updated, -1))
    out = pd.DataFrame(index=ticks)
    out["forecast_rain_class"] = np.where(best >= 0, best, np.nan)
    age = (ticks.to_numpy().astype("datetime64[ns]").astype("int64") - newest) / 60e9
    out["forecast_age_min"] = np.where(newest >= 0, age, np.nan)
    return out


def tide(path: str, ticks: pd.DatetimeIndex) -> pd.DataFrame:
    levels = pd.read_parquet(
        path, columns=["ts", "predicted_m", "minutes_to_high_tide"]
    )
    ts = pd.to_datetime(levels["ts"])
    if ts.dt.tz is None:  # Spark's instants, read back without their zone, are UTC
        ts = ts.dt.tz_localize("UTC")
    levels["ts"] = ts.dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    levels = levels.set_index("ts").sort_index()
    return levels.rename(columns={"predicted_m": "tide_height_m"}).reindex(ticks)


# --- the comparison ---------------------------------------------------------------------------------------------------


def ranking(fitted: gated.Fit) -> dict[str, float]:
    """Threshold-free out-of-fold scores of p_alert against the label: what the tide adds before any cut-off."""
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    y = fitted.rows["alert_label"].to_numpy()
    p = np.clip(fitted.p_alert, 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "base_rate": float(y.mean()),
    }


def matched(fitted: gated.Fit, share: float, judge: dict):
    """The scoreboard when the fit warns (Medium) on the `share` of development ticks with the highest p_alert (ties
    in time order), as one area."""
    k = round(share * len(fitted.p_alert))
    top = np.argsort(-fitted.p_alert, kind="stable")[:k]
    preds = fitted.rows.iloc[top][["area", "ts"]].assign(tier="Medium")
    board = score(preds, n_areas=1, **judge)
    m = board.metrics(ci=False)
    keep = ("alert_events", "alert_flagged_before_pub", "alert_mean_lead_min", "flood_events",
            "flood_flagged_before_pub", "warnings", "false_alarms_per_area_month", "warned_hours_per_area_month")  # fmt: skip
    return board, {k: m[k] for k in keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--landing", required=True)
    parser.add_argument(
        "--tide-levels", required=True, help="gold.tide_levels as parquet"
    )
    parser.add_argument(
        "--out", default=str(ROOT / "explorations" / "tide_ablation_local.json")
    )
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s")

    ticks = pd.date_range(
        pd.Timestamp(C.LABELS_START), END, freq=TICK, inclusive="left"
    )
    latest = posts(args.landing)
    alerts = alerts_from(latest)
    alerts = alerts[(alerts["alert_ts"] >= ticks[0]) & (alerts["alert_ts"] < END)]
    missing = missing_label_ticks(latest, ticks)
    log.info(
        "%d posts, %d alerts; %d of %d ticks with missing labels",
        len(latest),
        len(alerts),
        len(missing),
        len(ticks),
    )

    features = (
        rain(args.landing, ticks)
        .join(forecast(args.landing, ticks))
        .join(tide(args.tide_levels, ticks))
    )
    blind_mask = (features["rain_stamps"] < SLOTS) | features[
        "forecast_rain_class"
    ].isna()
    blind = pd.Series(ticks[blind_mask.to_numpy()])
    log.info("%d ticks with missing features", len(blind))

    rows = features.reset_index(names="ts").assign(area=AREA)
    rows["alert_label"] = tick_labels(rows, alerts)
    rows["flood_label"] = tick_labels(rows, alerts, kinds=[C.FLASH_FLOOD])
    keep = (
        ~rows["ts"].isin(missing)
        & ~blind_mask.to_numpy()
        & rows[list(C.MODEL_FEATURES)].notna().all(axis=1)
    )
    rows = rows[keep].reset_index(drop=True)
    rows["area"] = pd.Categorical(rows["area"], categories=[AREA])
    rows["split"] = "train"
    usable = gated.usable_alerts(alerts, rows)
    split = make_split(
        usable, start=ticks[0], test_start=END, test_end=END + pd.Timedelta(days=7)
    )
    log.info(
        "%d model rows, %d alert rows, %d usable alerts",
        len(rows),
        int(rows["alert_label"].sum()),
        len(usable),
    )

    judge = {
        "alerts": alerts,
        "start": split.start,
        "end": split.dev_end,
        "missing": missing,
        "blind": blind,
    }
    results = []
    for seed in SEEDS:
        fits = {}
        for name, drop in (("with", ()), ("without", C.FEATURE_GROUPS["tide"])):
            _, fitted = gated.fit(
                rows, alerts, split, missing=missing, blind=blind,
                settings=gated.Settings(seed=seed, drop=drop),
            )  # fmt: skip
            fits[name] = fitted
        result = {"seed": seed, "ranking": {}, "matched": {}, "paired": {}}
        for name, fitted in fits.items():
            result["ranking"][name] = {
                **ranking(fitted),
                "gate_pass_share": fitted.stats["gate_pass_share"],
            }
            result["matched"][name] = {
                f"{q:g}": matched(fitted, q, judge)[1] for q in SHARES
            }
            log.info("seed %d %s tide: %s", seed, name, result["ranking"][name])
        # the same PUB events, each fit warning on the same share of ticks
        result["paired"] = compare.paired_events(
            matched(fits["with"], PAIRED_SHARE, judge)[0],
            matched(fits["without"], PAIRED_SHARE, judge)[0],
        )
        result["tide_gain_share"] = compare.importance_share(
            fits["with"].importance, C.FEATURE_GROUPS["tide"]
        )
        result["importance_with"] = fits["with"].importance.to_dict("records")
        results.append(result)
    afternoon = rows[rows["ts"].dt.hour.between(12, 19)]
    summary = {
        "rows": len(rows),
        "alert_rows": int(rows["alert_label"].sum()),
        "alerts": len(alerts),
        "usable_alerts": len(usable),
        "development": f"{split.start.date()} to {split.dev_end.date()}",
        "missing_label_days": len(missing) * C.TICK_MINUTES / 1440,
        "blind_days": len(blind) * C.TICK_MINUTES / 1440,
        "alert_rate": float(rows["alert_label"].mean()),
        "alert_rate_by_tide_quartile": rows.groupby(
            pd.qcut(rows["tide_height_m"], 4), observed=True
        )["alert_label"]
        .mean()
        .rename(str)
        .to_dict(),
        # the same, within the afternoon storm hours only: does the tide's link to alerts survive the time of day?
        "alert_rate_by_tide_quartile_12_to_20h": afternoon.groupby(
            pd.qcut(afternoon["tide_height_m"], 4), observed=True
        )["alert_label"]
        .mean()
        .rename(str)
        .to_dict(),
        "results": results,
    }
    Path(args.out).write_text(
        json.dumps(
            summary,
            indent=1,
            default=lambda x: (
                None if isinstance(x, float) and math.isnan(x) else str(x)
            ),
        )
    )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
