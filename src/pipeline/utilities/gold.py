"""What the gold tables share: where bronze and silver are, and arithmetic on the 5-minute tick grid (the tick helpers
live in the floodsense package's features.py, which flood-live's score task shares).

Times are worked in epoch seconds, so no result depends on the session time zone. SGT has kept UTC+8 since 1982, so
an SGT midnight is the UTC one minus 8 hours, and the tick grid is the same in both.

Gold is built on a grid of 5-minute ticks, and most of its logic is deciding which ticks a thing counts at. The helpers
make that arithmetic exact. `tick_at_or_after`, `tick_after` and `tick_before` round an instant to the grid (the first
tick at or after it, strictly after it, strictly before it), and `ticks_between` expands a range of ticks, both ends
included, into one row per tick. A gold table says, for example, "this reading counts at the ticks from here to there",
rounds both ends, and explodes the range, which turns a time-window join into a plain group-by. `sgt_midnight_s` and
`sgt_day_floor` do the same for SGT days. `bronze` and `silver` give the qualified name of a table in the schemas set in
the pipeline's configuration.
"""

from datetime import date

from pyspark import cloudpickle
from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F

from daisi_team_master_corgis_floodsense import tides
from daisi_team_master_corgis_floodsense.features import (  # noqa: F401  (gold imports them from here)
    TICK_S,
    seconds,
    tick_after,
    tick_at_or_after,
    tick_before,
    ticks_between,
)

# The tide model's pandas functions run on the executors (applyInPandas), which can't import this package.
cloudpickle.register_pickle_by_value(tides)

DAY_S = 24 * 3600
SGT_S = 8 * 3600


def bronze(spark: SparkSession, name: str) -> str:
    return f"{spark.conf.get('bronze_schema')}.{name}"


def silver(spark: SparkSession, name: str) -> str:
    return f"{spark.conf.get('silver_schema')}.{name}"


def sgt_midnight_s(day: date) -> int:
    """Epoch seconds of `day`'s SGT midnight."""
    return (day - date(1970, 1, 1)).days * DAY_S - SGT_S


def sgt_day_floor(s: Column) -> Column:
    """The SGT midnight at or before epoch second `s`."""
    return F.floor((s + SGT_S) / DAY_S) * DAY_S - SGT_S
