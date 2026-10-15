"""The temporal split for the gated LightGBM pair.

The outer split (contract.py) trains on the first 80% of the recorded label time and tests once on the last 20%,
event by event. `temporal_split` chose where the test starts: the SGT midnight that leaves 20% of the recorded ticks
(gold.label_ticks) after it. Recorded time, not calendar time, because PUB's deleted posts leave months of missing
labels in 2021-2024 that count for neither side.

The pair learns four things from labels: the gate, the trees, the calibration of p_alert and the Medium/High
cut-offs. None of them may look at the test, so the development period is cut again into blocked folds:

- A fold is a run of whole SGT days. One storm crosses several areas and lasts hours, so splitting by area or by tick
  would put half a storm on each side.
- Each fold validates on its block and trains on the rest of the development period, minus `embargo_days` either side
  of the block. The embargo is at least the one-hour label horizon, so no training label reads an alert in the block.
  Raise it if a feature looks back more than a day.
- Blocks hold about the same number of alert days, and each cut sits midway through the dry gap between two alert days.
- The blocks tile the development period, so the folds' out-of-fold predictions cover every tick once. Calibration and
  the tier cut-offs are fitted on them, scored as one window by scoreboard.score. A single tail holdout would leave a
  handful of the training flood days to set the High cut-off on.

The same embargo separates development from the test: development ends `embargo_days` before the test, so nothing
fitted or tuned has a label that reads a test alert. The test ends at contract.SPLIT_END, the last midnight of the
label snapshot the split was computed on.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from . import contract as C
from .scoreboard import HORIZON

FOLDS = 5  # about 33 alert days and 4 flood days per block (Oct 2021 - Dec 2025, as placed on 15 Oct)
EMBARGO_DAYS = 1
_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Fold:
    """Validate on [valid_start, valid_end); train on the rest of [start, end) further than `embargo` from it."""

    start: pd.Timestamp
    end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    embargo: pd.Timedelta

    def train(self, ts: pd.Series) -> np.ndarray:
        """Mask over `ts`: the rows this fold's models fit on."""
        ts = _sgt(ts)
        apart = (ts < self.valid_start - self.embargo) | (
            ts >= self.valid_end + self.embargo
        )
        return (_within(ts, self.start, self.end) & apart).to_numpy()

    def valid(self, ts: pd.Series) -> np.ndarray:
        """Mask over `ts`: the rows this fold predicts out of fold."""
        return _within(_sgt(ts), self.valid_start, self.valid_end).to_numpy()


@dataclass(frozen=True)
class Split:
    """The development period [start, dev_end) cut into `folds`, and the test window [test_start, test_end)."""

    start: pd.Timestamp
    dev_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    folds: tuple[Fold, ...]

    def dev(self, ts: pd.Series) -> np.ndarray:
        """Mask over `ts`: the rows the final model refits on, once the folds have chosen its settings."""
        return _within(_sgt(ts), self.start, self.dev_end).to_numpy()

    def test(self, ts: pd.Series) -> np.ndarray:
        """Mask over `ts`: the rows scored once, at the end."""
        return _within(_sgt(ts), self.test_start, self.test_end).to_numpy()

    def cv(self, ts: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        """(train, valid) row positions per fold, as lightgbm.cv(folds=...) and scikit-learn take them."""
        return [
            (np.flatnonzero(f.train(ts)), np.flatnonzero(f.valid(ts)))
            for f in self.folds
        ]

    def summary(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """A row per fold and one for the test: days, alert days and flood days on each side. Log it with every run.

        The test row's train_* columns count the development period, which the final model refits on.
        """
        days = pd.Series(
            pd.date_range(self.start, self.test_end, freq="D", inclusive="left")
        )
        wet = days.isin(_alert_days(alerts)).to_numpy()
        floods = alerts[alerts["kind"] == C.FLASH_FLOOD]
        flooded = days.isin(_alert_days(floods)).to_numpy()

        def count(mask: np.ndarray, prefix: str = "") -> dict[str, int]:
            return {
                f"{prefix}days": int(mask.sum()),
                f"{prefix}alert_days": int((mask & wet).sum()),
                f"{prefix}flood_days": int((mask & flooded).sum()),
            }

        rows = [
            {
                "part": f"fold {i}",
                "first_day": f.valid_start,
                "last_day": f.valid_end - _DAY,
                **count(f.valid(days)),
                **count(f.train(days), "train_"),
            }
            for i, f in enumerate(self.folds, 1)
        ]
        rows.append(
            {
                "part": "test",
                "first_day": self.test_start,
                "last_day": self.test_end - _DAY,
                **count(self.test(days)),
                **count(self.dev(days), "train_"),
            }
        )
        return pd.DataFrame(rows)


def make_split(
    alerts: pd.DataFrame,
    *,
    labels_as_of=None,
    folds: int = FOLDS,
    embargo_days: int = EMBARGO_DAYS,
    start=C.TRAIN_START,
    test_start=C.TEST_START,
    test_end=C.SPLIT_END,
) -> Split:
    """Cut [start, test_start) into `folds` blocks with about the same number of alert days.

    `alerts` is silver.area_alerts. The test runs to `test_end`, or, given `labels_as_of` (when the labels were last
    refreshed), to the last midnight at least an hour before then if that comes sooner, so every test tick's label is
    complete.
    """
    start, test_start = pd.Timestamp(start), pd.Timestamp(test_start)
    embargo = pd.Timedelta(days=embargo_days)
    if embargo < HORIZON:
        raise ValueError(
            f"the embargo must cover the {C.HORIZON_MINUTES}-minute label horizon"
        )
    dev_end = test_start - embargo
    test_end = pd.Timestamp(test_end)
    if labels_as_of is not None:
        test_end = min(test_end, (pd.Timestamp(labels_as_of) - HORIZON).floor("D"))
    if test_end <= test_start:
        raise ValueError(f"labels as of {labels_as_of} complete no test day")

    days = pd.date_range(start, dev_end, freq="D", inclusive="left")
    wet = np.flatnonzero(days.isin(_alert_days(alerts)))
    if len(wet) < folds:
        raise ValueError(f"{len(wet)} alert days can't fill {folds} folds")

    cuts = [0]
    for i in range(1, folds):
        before = (i * len(wet) + folds // 2) // folds  # alert days left of the cut
        # the dry days between two alert days run from wet[before - 1] + 1 to wet[before] - 1; cut in the middle
        cuts.append((wet[before - 1] + 1 + wet[before]) // 2)
    bounds = [*days[cuts], dev_end]
    return Split(
        start=start,
        dev_end=dev_end,
        test_start=test_start,
        test_end=test_end,
        folds=tuple(Fold(start, dev_end, a, b, embargo) for a, b in pairwise(bounds)),
    )


def temporal_split(
    recorded: pd.Series,
    *,
    test_share: float = C.TEST_SHARE,
    embargo_days: int = EMBARGO_DAYS,
) -> pd.Timestamp:
    """The SGT midnight the test should start at: the one whose share of all recorded ticks, from it to the end, is
    closest to `test_share`. The embargo comes out of the development side.

    `recorded` holds one boolean per tick, indexed by the tick: True where the tick's label is known
    (gold.label_ticks, label_window = 'recorded'). Pass the whole period to split, start to end.
    """
    per_day = (
        recorded.astype(int)
        .groupby(_sgt(recorded.index.to_series()).dt.normalize().to_numpy())
        .sum()
    )
    total = per_day.sum()
    if not total:
        raise ValueError("no recorded ticks to split")
    after = per_day[::-1].cumsum()[::-1]  # recorded ticks from each day to the end
    best = (after / total - test_share).abs().idxmin()
    if best - pd.Timedelta(days=embargo_days) <= per_day.index[0]:
        raise ValueError("the test would leave no development period")
    return best


def _alert_days(alerts: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(_sgt(alerts["alert_ts"]).dropna().dt.normalize().unique())


def _sgt(ts: pd.Series) -> pd.Series:
    """Naive SGT, the convention the contract sets, so aware timestamps compare with the split's boundaries."""
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    return ts


def _within(t: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (t >= start) & (t < end)
