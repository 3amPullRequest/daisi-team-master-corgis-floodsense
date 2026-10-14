import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import contract as C
from daisi_team_master_corgis_floodsense.ml.split import make_split

# A one-month development period: March 2025, tested on April.
MARCH = {"start": "2025-03-01", "test_start": "2025-04-01"}


def day(d: str) -> pd.Timestamp:
    return pd.Timestamp(f"2025-{d}")


ALERTS = pd.DataFrame(
    [
        ("Bukit Timah", day("03-03 16:46"), C.WARNING),
        ("Bedok", day("03-04 15:10"), C.WARNING),
        ("Jurong West", day("03-10 17:02"), C.WARNING),
        ("Bukit Timah", day("03-15 16:20"), C.WARNING),
        ("Bukit Timah", day("03-15 16:45"), C.FLASH_FLOOD),
        ("Tampines", day("03-22 14:30"), C.WARNING),
        ("Ang Mo Kio", day("03-27 15:55"), C.FLASH_FLOOD),
        ("Bedok", day("03-31 18:00"), C.WARNING),  # in the embargo before the test
        ("Bukit Timah", day("04-02 16:00"), C.WARNING),
        ("Bukit Timah", day("04-02 16:30"), C.FLASH_FLOOD),
    ],
    columns=["area", "alert_ts", "kind"],
)


def split(**kw):
    return make_split(
        ALERTS, **{"labels_as_of": day("04-10 00:30"), "folds": 3} | MARCH | kw
    )


def ticks(first: str, last: str) -> pd.Series:
    return pd.Series(pd.date_range(day(first), day(last), freq="5min"))


def test_blocks_tile_development_and_cut_midway_through_dry_gaps():
    s = split()
    blocks = [(f.valid_start, f.valid_end) for f in s.folds]
    # 6 alert days, 2 per block. Dry 5-9 Mar: cut on the 7th. Dry 16-21 Mar: cut on the 19th.
    assert blocks == [
        (day("03-01"), day("03-07")),
        (day("03-07"), day("03-19")),
        (day("03-19"), day("03-31")),
    ]
    alert_days = s.summary(ALERTS).set_index("part")["alert_days"]
    assert alert_days.tolist() == [2, 2, 2, 1]  # then the test


def test_training_rows_stay_an_embargo_away_from_the_block():
    s = split()
    ts = ticks("03-01", "03-31 23:55")
    for f in s.folds:
        train = ts[f.train(ts)]
        # no training label (ts, ts + 1 h] reaches into the block, and no training row follows it within a day
        assert (
            (train < f.valid_start - pd.Timedelta(days=1))
            | (train >= f.valid_end + pd.Timedelta(days=1))
        ).all()
    # the middle block is 7-18 Mar, so its folds train up to 5 Mar and from 20 Mar
    edges = pd.Series(
        [day(d) for d in ("03-05 23:55", "03-06", "03-19 23:55", "03-20")]
    )
    assert s.folds[1].train(edges).tolist() == [True, False, False, True]


def test_out_of_fold_predictions_cover_every_development_tick_once():
    s = split()
    ts = ticks("03-01", "04-09 23:55")
    times_validated = sum(f.valid(ts).astype(int) for f in s.folds)
    assert (times_validated == s.dev(ts).astype(int)).all()
    assert s.dev(ts).sum() == 30 * 288  # 1-30 March


def test_development_ends_an_embargo_before_the_test_which_ends_at_complete_labels():
    s = split()
    assert s.dev_end == day("03-31") and s.test_start == day("04-01")
    # labels as of 00:30 on 10 Apr don't cover 9 Apr 23:55's next hour; as of 01:00 they do
    assert s.test_end == day("04-09")
    assert split(labels_as_of=day("04-10 01:00")).test_end == day("04-10")
    summary = split().summary(ALERTS).set_index("part")
    # 31 Mar's alert is in no fold, and not in what the final model refits on
    assert summary.loc["fold 3", "train_alert_days"] == 4
    test = summary.loc["test"]
    assert (test["days"], test["alert_days"], test["flood_days"]) == (8, 1, 1)
    assert (test["train_days"], test["train_alert_days"]) == (30, 6)


def test_summary_counts_both_sides_of_each_fold():
    summary = split().summary(ALERTS).set_index("part")
    first = summary.loc["fold 1"]
    assert first["days"] == 6 and first["flood_days"] == 0
    # trains on 8-30 March: everything after the block and its embargo day
    assert (
        first["train_days"],
        first["train_alert_days"],
        first["train_flood_days"],
    ) == (23, 4, 2)


def test_cv_gives_row_positions_in_the_frame_passed():
    s = split()
    gated = pd.DataFrame({"ts": ticks("03-01", "03-30 23:55")}).sample(
        frac=0.1, random_state=0
    )
    folds = s.cv(gated["ts"])
    assert len(folds) == 3
    for (train, valid), f in zip(folds, s.folds):
        assert not set(train) & set(valid)
        assert (
            gated["ts"]
            .iloc[valid]
            .between(f.valid_start, f.valid_end, inclusive="left")
            .all()
        )
    assert sorted(np.concatenate([v for _, v in folds])) == list(range(len(gated)))


def test_aware_timestamps_are_read_as_sgt():
    s = split()
    utc = pd.Series(
        pd.to_datetime(["2025-03-06 16:30"]).tz_localize("UTC")
    )  # 7 Mar 00:30 SGT
    assert s.folds[1].valid(utc).tolist() == [True]


def test_round_1_defaults():
    alerts = ALERTS.assign(alert_ts=ALERTS["alert_ts"] + pd.DateOffset(months=3))
    s = make_split(alerts, labels_as_of="2026-10-13 09:00", folds=3)
    assert (s.start, s.dev_end) == (
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-12-31"),
    )
    assert (s.test_start, s.test_end) == (
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-10-13"),
    )


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"folds": 7}, "6 alert days can't fill 7 folds"),
        ({"embargo_days": 0}, "label horizon"),
        ({"labels_as_of": day("04-01 12:00")}, "complete no test day"),
    ],
)
def test_refuses_splits_that_cannot_be_honest(kw, message):
    with pytest.raises(ValueError, match=message):
        split(**kw)
