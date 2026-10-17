"""What the gold tables share: where bronze and silver are, and arithmetic on the 5-minute tick grid.

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

from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F

from daisi_team_master_corgis_floodsense.ml import contract as C

TICK_S = C.TICK_MINUTES * 60
DAY_S = 24 * 3600
SGT_S = 8 * 3600


def bronze(spark: SparkSession, name: str) -> str:
    return f"{spark.conf.get('bronze_schema')}.{name}"


def silver(spark: SparkSession, name: str) -> str:
    return f"{spark.conf.get('silver_schema')}.{name}"


def seconds(ts: str) -> Column:
    """Epoch seconds of a TIMESTAMP column."""
    return F.unix_timestamp(ts)


def sgt_midnight_s(day: date) -> int:
    """Epoch seconds of `day`'s SGT midnight."""
    return (day - date(1970, 1, 1)).days * DAY_S - SGT_S


def sgt_day_floor(s: Column) -> Column:
    """The SGT midnight at or before epoch second `s`."""
    return F.floor((s + SGT_S) / DAY_S) * DAY_S - SGT_S


def tick_at_or_after(s: Column) -> Column:
    return F.ceil(s / TICK_S) * TICK_S


def tick_after(s: Column) -> Column:
    return F.floor(s / TICK_S) * TICK_S + TICK_S


def tick_before(s: Column) -> Column:
    return F.ceil(s / TICK_S) * TICK_S - TICK_S


def ticks_between(first: Column, last: Column) -> Column:
    """One row per tick from `first` to `last` (epoch seconds on the grid, both included); none if last < first."""
    step = F.lit(TICK_S).cast("bigint")
    return F.explode(
        F.when(
            last >= first, F.sequence(first.cast("bigint"), last.cast("bigint"), step)
        )
    )
