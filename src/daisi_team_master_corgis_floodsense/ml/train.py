"""Train the FloodSense models and the rainfall rule on gold, log them to MLflow, and register the best in Unity Catalog.
The flood-train job runs it on serverless after refreshing gold (resources/train.job.yml):

    databricks bundle run train -t dev -p flood_dev
    databricks bundle run train -t dev -p flood_dev --params score_test=true   # the single look at the test

What it reads, every value point-in-time and none of it from a missing interval:
- gold.model_rows, split 'train': the area-ticks whose label window and feature window are both recorded;
- gold.label_ticks and gold.feature_ticks: which ticks' labels or features are missing, for the scoreboard;
- silver.area_alerts: PUB's alerts, which the scoreboard judges against.

The models (--models), each a spec `family[:option]...`, comma-separated:

- `ordinal` (the default since 23 Oct, ordinal.py): two rungs, the flood rung on every row. Options `alert=` and
  `flood=` (`lightgbm` or `logistic`, each rung's learner; `alert=gated` makes the alert rung the gated pair, its gate
  focusing the rater without vetoing a warning, at `gate_recall=`) and `flood_weight=`.
- `gated` (gated.py, the registered versions up to v5, and v10 on): the gate and the rater. Options `gate_recall=`,
  `area_effect=` (`credibility` or `category`), `flood_weight=`, `half_life=` (the credibility prior's recency
  half-life in days, credibility.recency; none by default, as v5), and `medium=` and `high=`: the tiers as fixed chances
  of a PUB alert within the hour (v11 and final_1 have `medium=0.01:high=0.05`, final_2 `medium=0.03:high=0.05`),
  instead of Medium chosen within the rule's false-alarm rate and High by its flood precision, as sweeps compare fits.
- either family reads `base` features by default (contract.BASE_FEATURES, what the gated pair v5 reads) or, with
  `extended`, contract.EXTENDED_FEATURES (the dew point for the humidity, and the prior rain); `drop=` leaves out
  feature groups (contract.FEATURE_GROUPS) or single model features, joined by +, and `add=` reads them beyond the set
  (`add=pub_alerts`: PUB's alerts just posted nearby; `add=storm`, `add=surplus`, `add=forecast_groups`: the storm so
  far and the forecast in fewer groups, 24 Oct; `add=map_line`, `add=map_polygon`: the area's ground from the national
  map's lines and polygons, 25 Oct; none of them in either set). `seed=` refits with another seed (the
  quiet rows' sample and LightGBM's), which shows how far a refit alone moves the scores. E.g.
  `gated,gated:extended,gated:add=pub_alerts,gated:half_life=730,ordinal:alert=gated:flood=logistic`,
  `gated:drop=forecast_rain_class:add=forecast_rain_group4`.

The default is `gated`: on 23 Oct, out of fold, nothing beat it on the ranking below (ARCHITECTURE.md). (That day's runs
named the feature sets `legacy` for base and `current` for extended.)

What it logs to one MLflow run: the split and its folds; how much of each split the models can use; the rule's fit;
each model's fit (a nested run per spec, `sweep.json` setting them side by side); the chosen one scored by
scoreboard.score over the development period out of fold, as the rule is; and whether it beats the rule: more lead time
at no more false alarms, judged at Medium when its cut-off is held to the rule's false-alarm rate, and at High
(oof_high_*) when the tiers are fixed chances, as Medium is then a heads-up that warns far more often than the rule;
with how often each tier's warnings came true (oof_medium_warning_alert_share, oof_high_warning_alert_share). The
chosen model is registered as
@challenger, and as @champion as well when it beats the rule, unless --promote false leaves that to flood-verify
(verify.py), which checks the registered model first; --register false logs it without registering, for experiments.
It also logs each development flood one by one, out of fold (oof_floods.json, floods.flood_table): its lead time, and
the tier and chance the model showed before it; and for a gated pair, tiers.json (diagnostics.tier_table): Medium and
High at other cut-offs, and High where the rule's rain is falling, each scored as the pair's own tiers are.

The sweep: each spec is a nested run, and `sweep.json` sets them side by side: the rows warned and the share of alert
and flood rows among them, average precision for alerts and for floods (PR-AUC, against each base rate), the cut-offs,
the out-of-fold events flagged (a flood counting --flood-value times, as `_rank` ranks the fits and each fit chooses
its Medium cut-off), lead and false alarms, and the best each fit flags at 2 and 4 false alarms per area-month, which
shows how much the false-alarm budget alone holds a model back. Every fit after the first is paired with the first, on
the same events and rows (`versus_first`, compare.py): the difference in rank, in PUB events and in floods flagged
before PUB (the events only one fit flagged, with an exact McNemar test), and in every row's log loss, each with a 95%
interval and p-value from a bootstrap over whole days (vs_first_* in sweep.json and in the fit's nested run, whose
oof_events.json lists its events). So put the reference first, and refit it with seed= to see how far a refit alone
moves them. Each fit also gets diag_* (diagnostics.py): rows and warning episodes confused at Medium and at High, PUB
events missed or warned late, the chances' quantiles, skewness, ROC AUC and calibration, and the events its Medium curve
flags at budgets from 0.5 to 4 false alarms; and its reliability.json. --flood-weight is what a flood row weighs in
training
where a spec doesn't say (1 by default: 10 cost the gated pair 12 points of events and caught no more floods, 22 Oct);
--flood-value what a flood flagged is worth in choosing (scoreboard.FLOOD_EVENT_VALUE).

With --override true (the default), the rainfall rule is tried as every fit's safety override, of either family
(ordinal.add_override): a threshold on the rule's window above which a tick is at least Medium, chosen with the Medium
cut-off out of fold within the same budget, and kept only if it flags more (override.json in each nested run; sweep.json
has each fit's choice, and its scores before it as plain_*). Fits are ranked with their overrides.

With --compare-without (default 'tide'), it refits the chosen model without each named group of
contract.FEATURE_GROUPS (`antecedent` is the prior rain), in a nested run per group, and logs what the group adds: the
headline metrics side by side, the PUB events matched one to one, and the group's share of the models' gain
(compare.py).

With --score-test true it also scores the test split, once, for both models: metrics with 95% CIs, the event-by-event
tables, and the test predictions in ml.predictions. It refuses while the test's features haven't landed.

The order of work in `main`:

1. Load. The training rows come into pandas compactly (`load_rows`: area as a category, features as float32, labels as
   int8) so ten million of them fit on a serverless driver. The alerts, the ticks with missing labels and the ticks
   with missing features come with them, and so does a coverage table of how much of each split the models can use.
2. Split. `make_split` cuts the development period into folds from the alerts the rows can actually learn from
   (`gated.usable_alerts`), so no fold is made of days without rows.
3. Fit the rule first. Its false-alarm rate sets the budget each model's Medium cut-off may use, so that the comparison
   asks for more lead time at no more false alarms.
4. Fit each spec, a nested MLflow run scored out of fold by the same scoreboard as the rule, with the override if it
   earns its place. The fit that flags the most events before PUB wins (a flood event counting --flood-value times),
   then the longer lead, then fewer false alarms (`_rank`).
5. Judge and register. The verdict is logged, the winner is logged as a pyfunc model (`log_model`, with serving.py and
   this package's source) and, unless --model is empty, registered. A registered model gets the challenger alias, and
   the champion alias too if it beats the rule (and --promote isn't false).
6. The test, only on request. `check_test_coverage` refuses before any fitting if too little of the test has features.
   `score_test` writes the predictions to ml.predictions before it logs a single test metric, so that if the write
   fails nobody has seen the test, and the rerun is still the first look.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, compare, diagnostics, floods, gated, ordinal
from . import contract as C
from .scoreboard import FLOOD_EVENT_VALUE, score
from .split import make_split

log = logging.getLogger(__name__)

GATED, ORDINAL = "gated", "ordinal"
# what each family's spec options are, and how to read them
SPEC_OPTIONS = {
    GATED: {
        "gate_recall": float, "area_effect": str, "flood_weight": float, "half_life": float, "drop": str, "add": str,
        "seed": int, "medium": float, "high": float,
    },
    ORDINAL: {
        "alert": str, "flood": str, "flood_weight": float, "gate_recall": float, "drop": str, "add": str, "seed": int,
    },
}  # fmt: skip
# the settings field an option sets, where its name differs
SPEC_FIELDS = {
    "alert": "alert_learner",
    "flood": "flood_learner",
    "half_life": "prior_half_life_days",
    "medium": "medium_chance",
    "high": "high_chance",
}


@dataclasses.dataclass(frozen=True)
class Spec:
    """One model to fit: the text asking for it (`name`), its family and its options (module docstring)."""

    name: str
    family: str
    options: tuple[tuple[str, object], ...] = ()

    @property
    def module(self):
        return gated if self.family == GATED else ordinal

    @property
    def prefix(self) -> str:
        """The prefix of its stats in MLflow: pair_* for a gated pair (as verify.py reads v1-v5), model_* else."""
        return "pair" if self.family == GATED else "model"

    @property
    def features(self) -> str:
        return dict(self.options).get("features", ordinal.BASE)

    def settings(self, **defaults):
        """The fit's settings: the family's defaults, then `defaults` (the run's budget, flood weight and value), then
        the spec's own options."""
        options = {SPEC_FIELDS.get(k, k): v for k, v in self.options if k not in ("features", "drop", "add")}
        # drop= and add=: feature groups (contract.FEATURE_GROUPS) or model features to leave out of the feature set or
        # read beyond it, joined by +, e.g. drop=antecedent, add=pub_alerts, drop=forecast_rain_class
        dropped, added = (self._group_features(key) for key in ("drop", "add"))
        if self.family == ORDINAL:
            return ordinal.Settings(
                **{**defaults, "features": self.features, "drop": dropped, "add": added, **options}
            )
        read = (*ordinal.FEATURE_SETS[self.features], *added)
        outside = tuple(c for c in C.MODEL_FEATURES if c not in read)
        return gated.Settings(**{**defaults, "drop": tuple(dict.fromkeys((*outside, *dropped))), **options})

    def _group_features(self, key: str) -> tuple[str, ...]:
        names = [g for g in str(dict(self.options).get(key, "")).split("+") if g]
        unknown = set(names) - set(C.FEATURE_GROUPS) - set(C.MODEL_FEATURES)
        if unknown:
            raise ValueError(
                f"no feature group or model feature {sorted(unknown)}: {key}= takes {', '.join(C.FEATURE_GROUPS)} "
                "or a model feature"
            )
        return tuple(dict.fromkeys(c for g in names for c in C.FEATURE_GROUPS.get(g, (g,))))

    def params(self, settings) -> dict:
        """What MLflow logs of the fit's settings."""
        keys = (
            ("gate_recall", "area_effect", "prior_half_life_days", "medium_chance", "high_chance")
            if self.family == GATED
            else ("alert_learner", "flood_learner", "gate_recall", "storm_quiet_share", "dry_quiet_share")
        )
        return {
            "model_spec": self.name,
            "family": self.family,
            "features": self.features,
            "dropped": ",".join(settings.drop),
            "flood_weight": settings.flood_weight,
            "seed": settings.seed,
            **{k: getattr(settings, k) for k in keys},
        }

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
# what each fit of a sweep is paired against: the first one (vs_first_* in sweep.json and in its nested run)
VERSUS = "vs_first"
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
    with mlflow.start_run(run_name=f"flood-train {datetime.now():%Y-%m-%d %H:%M}"):
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
        # every fit reads the development rows as sorted here (development_rows passes them on uncopied), so the
        # training frame itself can go: ten million rows are 1.4 GB, and a sweep ran out of memory on 23 Oct
        del rows
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
        # Round 1: the model must beat the rule on lead time and false alarms. So its Medium cut-off may raise no more
        # false alarms than the rule does, and the verdict compares lead at that matched rate (with the tiers as fixed
        # chances, Medium isn't held to it, and the verdict reads High instead).
        budget = min(
            gated.DEFAULTS.max_false_alarms, rule_metrics["false_alarms_per_area_month"]
        )
        mlflow.log_param("false_alarm_budget", round(budget, 4))
        defaults = {
            "max_false_alarms": budget,
            "flood_weight": args.flood_weight,
            "flood_value": args.flood_value,
        }

        # only the best fit so far is kept, so a long sweep holds no more than two fits of ten million rows at once (and
        # the first fit's events and chances, which the others are paired with)
        best, sweep, first = None, [], None
        for spec in args.models:
            settings = spec.settings(**defaults)
            with mlflow.start_run(run_name=spec.name, nested=True):
                model, fitted = spec.module.fit(
                    dev, alerts, split, missing=missing, blind=blind, settings=settings
                )
                board = score(fitted.predictions(), **judge)
                metrics = board.metrics()
                # what each feature group adds is measured on a fit as it came, before any override
                plain = (fitted, board, metrics)
                mlflow.log_params(spec.params(settings))
                _log_metrics("plain_oof", metrics)
                mlflow.log_table(fitted.curve, "medium_curve.json")
                mlflow.log_table(fitted.importance, "importance.json")
                if fitted.area_prior is not None:
                    mlflow.log_table(fitted.area_prior, "area_prior.json")
                if args.override:
                    # every fit gets its own best override (none, unless it flags more), so fits are compared each
                    # with its safety net
                    model, fitted, overrides = ordinal.add_override(
                        model, fitted, dev, alerts, split, rule,
                        missing=missing, blind=blind, settings=settings,
                    )  # fmt: skip
                    mlflow.log_table(overrides, "override.json")
                    if model.override is not None:
                        board = score(fitted.predictions(), **judge)
                        metrics = board.metrics()
                _log_metrics(spec.prefix, fitted.stats)
                _log_metrics("oof", metrics)
                mlflow.log_table(_dated(board.events), "oof_events.json")
                mlflow.log_table(fitted.reliability(), "reliability.json")
                # beyond the headline: rows and warnings confused at each tier, the chances' shape and calibration
                diag = diagnostics.summary(fitted, board)
                _log_metrics("diag", diag)
                if first is None:
                    first, versus = Reference.of(fitted, board), {}
                else:
                    versus = versus_first(first, fitted, board, args.flood_value)
                    _log_metrics(VERSUS, versus)
            row = sweep_row(spec, settings, fitted, metrics, args.flood_value)
            row["override_mm"] = fitted.stats.get("override_mm", math.nan)
            row["plain_alert_flagged_before_pub"] = plain[2]["alert_flagged_before_pub"]
            row["plain_flood_flagged_before_pub"] = plain[2]["flood_flagged_before_pub"]
            row |= {f"{VERSUS}_{k}": float(v) for k, v in versus.items()}
            row |= {f"diag_{k}": float(v) for k, v in diag.items()}
            sweep.append(row)
            log.info("%s: %s", spec.name, row)
            rank = _rank(metrics, args.flood_value)
            if best is None or rank > best[0]:
                best = (rank, spec, settings, model, fitted, board, metrics, plain)
            del model, fitted, board, plain
            gc.collect()
        _, spec, chosen, model, fitted, board, oof, plain = best
        mlflow.log_table(pd.DataFrame(sweep), "sweep.json")
        log.info("chosen: %s, override %s", spec.name, model.override or "none")

        mlflow.log_params(
            {**spec.params(chosen), "flood_value": chosen.flood_value}
        )
        _log_metrics(spec.prefix, fitted.stats)
        _log_metrics("oof", oof)
        mlflow.log_table(fitted.curve, "medium_curve.json")
        mlflow.log_table(fitted.reliability(), "reliability.json")
        mlflow.log_table(fitted.importance, "importance.json")
        mlflow.log_table(_dated(board.events), "oof_events.json")
        if fitted.area_prior is not None:
            mlflow.log_table(fitted.area_prior, "area_prior.json")
        # each development flood one by one, out of fold: its lead time, and the tier and chance before it
        flood_rows = floods.flood_table(fitted.predictions(warned_only=False), **judge)
        mlflow.log_table(_dated(flood_rows), "oof_floods.json")
        _log_metrics("oof", floods.summary(flood_rows))
        if spec.family == GATED:
            # other ways to set a pair's tiers on its p_alert, scored as its own are (an ordinal model's tiers read its
            # warning score instead): Medium and High at other cut-offs, and High where the rule's rain is falling
            rain = dev.loc[fitted.rows.index, rule.column].to_numpy()
            tiers = diagnostics.tier_table(
                fitted, alerts, split, medium_p=model.medium_p, missing=missing, blind=blind,
                high_where={f"{rule.column} >= {rule.medium_mm:g} (the rule)": rain >= rule.medium_mm},
            )  # fmt: skip
            mlflow.log_table(tiers, "tiers.json")
        # How often each tier's warnings came true, for the app; and the tier the verdict judges: the one that warns at
        # about the rule's rate. That is Medium when it was chosen within that rate. With the tiers as fixed chances,
        # Medium is a heads-up that warns far more often, and High is the like-for-like warning (oof_high_*).
        _log_metrics("oof", {"medium_warning_alert_share": _followed_share(board)})
        judged_at, verdict_tier = oof, "Medium"
        if getattr(chosen, "medium_chance", None) is not None and math.isfinite(model.high_p):
            high_board = score(_high_only(fitted.predictions()), **judge)
            judged_at, verdict_tier = high_board.metrics(), "High"
            _log_metrics("oof_high", {**judged_at, "warning_alert_share": _followed_share(high_board)})
        mlflow.log_param("verdict_tier", verdict_tier)
        verdict = {
            # lead before PUB's first alert, misses counting 0, over every judged PUB event
            "beats_rule_lead": judged_at["alert_mean_lead_min"]
            > rule_metrics["alert_mean_lead_min"],
            "within_rule_false_alarms": judged_at["false_alarms_per_area_month"]
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
            "out of fold: %s flags %.1f%% of events before PUB (rule %.1f%%), lead %.1f min (rule %.1f), "
            "%.2f false alarms per area-month (rule %.2f)",
            spec.name,
            100 * oof["alert_flagged_before_pub"],
            100 * rule_metrics["alert_flagged_before_pub"],
            oof["alert_mean_lead_min"],
            rule_metrics["alert_mean_lead_min"],
            oof["false_alarms_per_area_month"],
            rule_metrics["false_alarms_per_area_month"],
        )

        for group in args.compare_without:
            compare_without(
                group, dev, alerts, split, judge, missing, blind, spec, chosen, *plain
            )
            gc.collect()

        full_name = (
            f"{args.catalog}.{args.ml_schema}.{args.model}"
            if args.model and args.register
            else None
        )
        info = log_model(model, dev, fitted, full_name)
        version = info.registered_model_version if full_name else None
        if version is not None:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(full_name, C.CHALLENGER, version)
            promoted = beats_rule and args.promote
            if promoted:
                client.set_registered_model_alias(full_name, C.CHAMPION, version)
            log.info(
                "registered %s version %s as @%s%s",
                full_name,
                version,
                C.CHALLENGER,
                f" and @{C.CHAMPION}" if promoted else "",
            )

        if args.score_test:
            name = (
                f"{args.model or C.REGISTERED_MODEL} v{version}"
                if version
                else model.name
            )
            score_test(
                spark,
                gold,
                args,
                areas,
                coverage,
                split,
                model,
                name,
                rule,
                alerts,
                missing,
                blind,
            )
    return 0


@dataclasses.dataclass(frozen=True)
class Reference:
    """What the sweep's first fit leaves for the others to be paired with: its scoreboard, its out-of-fold chance of
    every development row (float32, so ten million rows cost 40 MB), its stats, and a sample of its rows' ticks to check
    that a later fit's rows are the same ones in the same order."""

    board: object
    p_alert: np.ndarray
    stats: dict
    rows: int
    ticks: np.ndarray

    @classmethod
    def of(cls, fitted, board) -> Reference:
        return cls(
            board, fitted.p_alert.astype(np.float32), dict(fitted.stats), len(fitted.rows), _tick_sample(fitted.rows)
        )


def versus_first(first: Reference, fitted, board, flood_value: float) -> dict[str, float]:
    """A fit paired with the sweep's first on the same development events and rows, out of fold (compare.py), each
    difference this fit's minus the first's: the rank (a flood counting `flood_value` times), the PUB events and the
    floods flagged before PUB, every row's log loss, and the average precision."""
    rows = fitted.rows
    if len(rows) != first.rows or not np.array_equal(_tick_sample(rows), first.ticks):
        raise ValueError("a fit's development rows differ from the first fit's: they can't be paired")
    return {
        **compare.paired_rank(board, first.board, flood_value),
        **compare.paired_events(board, first.board, "alert"),
        **compare.paired_events(board, first.board, "flood"),
        **compare.paired_rows(fitted.p_alert, first.p_alert, rows["alert_label"].to_numpy(), rows["ts"]),
        **{
            f"{k}_difference": fitted.stats.get(k, math.nan) - first.stats.get(k, math.nan)
            for k in ("alert_pr_auc", "flood_pr_auc", "brier")
        },
    }


def _high_only(preds: pd.DataFrame) -> pd.DataFrame:
    """A fit's High ticks as the only warnings, so the scoreboard judges High as it judges Medium."""
    return preds[preds["tier"] == "High"].assign(tier="Medium")


def _followed_share(board) -> float:
    """The share of a scoreboard's judged Medium warnings (Medium or above) that a PUB alert followed."""
    d = diagnostics.episodes_and_events(board)
    return d["medium_episode_hits"] / d["medium_episodes"] if d["medium_episodes"] else math.nan


def _tick_sample(rows: pd.DataFrame) -> np.ndarray:
    return rows["ts"].to_numpy()[::997]


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
    spec,
    chosen,
    fitted,
    board,
    oof,
) -> pd.DataFrame:
    """Refit the chosen model without one feature group, with the chosen fit's settings (the false-alarm budget, flood
    weight and the rest), in a nested run, and log the comparison to the parent run (metrics prefixed
    without_<group>_)."""
    import mlflow

    drop = C.FEATURE_GROUPS[group]
    with mlflow.start_run(run_name=f"without {group}", nested=True):
        settings = dataclasses.replace(chosen, drop=tuple(dict.fromkeys((*chosen.drop, *drop))))
        _, without = spec.module.fit(
            rows,
            alerts,
            split,
            missing=missing,
            blind=blind,
            settings=settings,
        )
        without_board = score(without.predictions(), **judge)
        without_oof = without_board.metrics()
        mlflow.log_params({**spec.params(settings), "dropped": ",".join(drop)})
        _log_metrics(spec.prefix, without.stats)
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


def log_model(model, dev: pd.DataFrame, fitted, full_name: str | None):
    """Log the model as a pyfunc model (with this package, so loading needs no install) and register it if named: an
    ordinal model as FloodSenseModel with its files in the `model` artifact, a gated pair as GatedPairModel in `pair`,
    as versions 1 to 5 were."""
    import lightgbm
    import mlflow
    from mlflow.models import infer_signature

    from .serving import FloodSenseModel, GatedPairModel

    pair = isinstance(model, gated.GatedPair)
    artifact, python_model = ("pair", GatedPairModel()) if pair else ("model", FloodSenseModel())
    # an input example with warned rows and quiet ones, so every output column has values to type
    order = np.argsort(-fitted.p_alert)
    picked = np.concatenate([order[:3], order[-2:]])
    example = dev.iloc[picked][["area", "ts", *C.MODEL_FEATURES]].astype(
        {"area": str, **{c: "float64" for c in C.MODEL_FEATURES}}
    )
    example = example.reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / artifact
        model.save(saved)
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            artifacts={artifact: str(saved)},
            code_paths=[str(PACKAGE)],
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            pip_requirements=[
                f"lightgbm=={lightgbm.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
            ],
            registered_model_name=full_name,
        )


def score_test(
    spark, gold, args, areas, coverage, split, model, name, rule, alerts, missing, blind
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
    model_preds = model.predict(test).assign(model=name)
    rule_preds = rule.predict(test)
    results = {}
    for prefix, preds in (("test", model_preds), ("rule_test", rule_preds)):
        board = score(_warned(preds), **window)
        results[prefix] = (board.metrics(), board.events)
    # Nothing about the test is logged until its predictions are written: if the write fails, nobody has seen the
    # test, and the rerun is still the first look.
    write_predictions(
        spark,
        f"{args.catalog}.{args.ml_schema}.{C.PREDICTIONS}",
        pd.concat([model_preds, rule_preds]),
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
        "--models",
        type=_specs,
        default=GATED,
        help="comma-separated models to fit, each family[:option]... (module docstring); the out-of-fold scoreboard "
        "picks one (default %(default)s)",
    )
    parser.add_argument(
        "--flood-weight",
        type=float,
        default=gated.DEFAULTS.flood_weight,
        help="what a row a flash flood followed weighs against a warning's in training, where a model's spec doesn't "
        "say (default %(default)s)",
    )
    parser.add_argument(
        "--flood-value",
        type=float,
        default=FLOOD_EVENT_VALUE,
        help="what a flood flagged before PUB counts against another event, in ranking the fits and choosing each "
        "one's Medium cut-off (default %(default)s)",
    )
    parser.add_argument(
        "--override",
        type=lambda text: text.strip().lower() != "false",
        default=True,
        help="'false' not to try the rainfall rule as each fit's safety override",
    )
    parser.add_argument(
        "--promote",
        type=lambda text: text.strip().lower() != "false",
        default=True,
        help="'false' to register the model as @challenger only, even if it beats the rule, so flood-verify can check "
        "it before it goes live",
    )
    parser.add_argument(
        "--register",
        type=lambda text: text.strip().lower() != "false",
        default=True,
        help="'false' to log the chosen model without registering it (experiments)",
    )
    parser.add_argument(
        "--compare-without",
        type=lambda text: [g.strip() for g in text.split(",") if g.strip()],
        default=["tide"],
        help=f"comma-separated feature groups ({', '.join(C.FEATURE_GROUPS)}) to refit the model without and "
        "compare; "
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


def _specs(text: str) -> list[Spec]:
    """'gated,ordinal:flood=logistic,gated:extended' as Specs, each checked now (its settings built) rather than an hour
    into the run."""
    out = []
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        family, *options = [o.strip() for o in part.split(":")]
        if family not in SPEC_OPTIONS:
            raise argparse.ArgumentTypeError(f"a model is one of {tuple(SPEC_OPTIONS)}, not {family!r} ({part!r})")
        read = []
        for option in options:
            if option in ordinal.FEATURE_SETS:
                read.append(("features", option))
                continue
            key, sep, value = option.partition("=")
            if not sep or key not in SPEC_OPTIONS[family]:
                raise argparse.ArgumentTypeError(
                    f"{family} takes {', '.join(SPEC_OPTIONS[family])} (key=value) or a feature set "
                    f"({', '.join(ordinal.FEATURE_SETS)}), not {option!r}"
                )
            read.append((key, SPEC_OPTIONS[family][key](value)))
        spec = Spec(part, family, tuple(read))
        try:
            spec.settings()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{part!r}: {exc}") from exc
        out.append(spec)
    if not out:
        raise argparse.ArgumentTypeError("name at least one model to fit")
    return out


def sweep_row(spec: Spec, settings, fitted, metrics: dict, flood_weight: float) -> dict[str, float]:
    """One fit of the sweep, for sweep.json: its spec and settings, the rows it warns (and its gate's pass share, for a
    pair), its average precision, its cut-offs, its out-of-fold scoreboard, and the best its Medium curve flags at the
    looser false-alarm budgets."""
    s = fitted.stats
    row = {
        "model_spec": spec.name,
        "family": spec.family,
        "features": spec.features,
        "flood_weight": settings.flood_weight,
        **{
            k: s.get(k, math.nan)
            for k in (
                "warned_row_share",
                "alert_row_recall",
                "flood_row_recall",
                "alert_pr_auc",
                "flood_pr_auc",
                "alert_base_rate",
                "flood_base_rate",
                "gate_pass_share",
                "gate_alert_row_recall",
                "gate_flood_row_recall",
                "gate_rounds",
                "rater_rounds",
                "alert_rounds",
                "flood_rounds",
                "alert_penalty",
                "flood_penalty",
                "medium_p",
                "high_p",
                "medium_cut",
                "high_cut",
                "brier",
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
                "flood_mean_lead_min",
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
