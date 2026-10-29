"""Each area's prior: how alert-prone its own history says it is, by Bühlmann-Straub credibility. Both models of the
gated pair start from it (an offset, LightGBM's init_score) in place of reading the area as a category.

Why not the category. In the training rows of 21 Oct, 15 of the 47 areas had no PUB alert and the rest from 2 to 61
alert days. LightGBM fits a category's own rate with next to no smoothing (cat_smooth is 20 rows against 400,000 an
area), so an area without an alert is pushed towards a zero chance whatever the weather. Its quiet rows also pull the
shared rain effects down everywhere early in boosting, when they are most of the gradient. And an area with a few
warnings sat so low that 85 mm of rain in an hour under a heavy-thundery forecast rated 0.2% (Lim Chu Kang, 30 Sep
2026, a flash flood 8 minutes later). Ten of the fifteen areas have no PUB water-level sensor, so their zero is real for
warnings; the other five have sensors and simply haven't alerted yet. Credibility weighs the two from the data: it
trusts an area's own history as far as its exposure and the real spread between areas warrant, and falls back on the
island for the rest. As an offset it also leaves the trees to learn the weather on top of the prior, and an area with a
low prior adds little gradient, so its quiet rows no longer drag the rain effects down in the areas that do flood.

The observation is an area-day (`area_days`), since alerts come in storms and one storm is many alert ticks:

- its claim is 1 for a day with a PUB alert in the area, `flood_weight` for a day with a flash flood, else 0, so a
  flash flood weighs as much in the prior as in the models' training;
- its exposure is the share of the day's ticks recorded for the area (gold.model_rows rows / 288), so missing time
  counts for nothing.

Then, per Bühlmann and Straub, with m_ad the exposure and X_ad = c_ad / m_ad the claim rate of area a on day d, m_a and
X_a the area's total exposure and rate, m and X the island's:

- process variance (within an area): s2 = sum_a sum_d m_ad (X_ad - X_a)^2 / sum_a (n_a - 1);
- variance of the areas' true rates: v = (sum_a m_a (X_a - X)^2 - (A - 1) s2) / (m - sum_a m_a^2 / m);
- credibility: Z_a = m_a / (m_a + s2 / v), and Z_a = 0 for every area if v <= 0 (the spread is all noise);
- prior rate: P_a = Z_a X_a + (1 - Z_a) X_Z, with X_Z the Z-weighted mean of the X_a (the estimator that keeps the
  total right);
- offset: log(P_a / X), the area's log relative risk. An area the prior hasn't seen gets 0, the island's.

gated.fit computes a prior inside each fold from that fold's training days only, so no out-of-fold score reads its own
block's alerts, and one from the whole development period for the final models.

Recency (23 Oct, `half_life_days`, off by default). PUB's alerts move between sensor sites every year: 25-43% of a
year's site alert-days come from sites that never alerted before (2023-2026). So a prior can weigh each area-day's
exposure and claim by 0.5 ** (days from a reference day / half_life_days), which makes old years count less without
changing a day's rate. The final models' prior weighs back from its last day. A fold's prior weighs both ways from the
middle of the block it predicts (`two_sided`), the out-of-fold stand-in for "the years nearest the one scored"; live,
only the past exists, so it is a little kinder than the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contract as C
from .split import _sgt

TICKS_PER_DAY = 24 * 60 // C.TICK_MINUTES
# a prior is never put below this share of the island's rate: only a guard against log(0), as Z < 1 always leaves an
# area some of the island's rate
MIN_RELATIVE_RISK = 1e-3


@dataclass(frozen=True)
class AreaPrior:
    """`table`: one row per area (exposure_days, claims, rate, credibility, prior_rate, offset); `k` = s2 / v, the
    exposure in days at which an area's own history gets half the weight (inf when the areas don't differ beyond noise);
    `island_rate`: claims per recorded area-day."""

    table: pd.DataFrame
    k: float
    island_rate: float

    @property
    def offsets(self) -> dict[str, float]:
        return dict(zip(self.table["area"], self.table["offset"].astype(float)))

    def offset(self, area: pd.Series) -> np.ndarray:
        """The offset of each row's area; 0 for an area the prior hasn't seen."""
        return area_offsets(area, self.offsets)


def area_offsets(area: pd.Series, offsets: dict[str, float]) -> np.ndarray:
    """Per row, its area's offset from `offsets` (0 for an area not in it). Fast on a category column, which is how
    train.load_rows reads the area."""
    area = pd.Series(area)
    if isinstance(area.dtype, pd.CategoricalDtype):
        by_code = np.array(
            [offsets.get(a, 0.0) for a in area.cat.categories] + [0.0], dtype=float
        )
        return by_code[area.cat.codes.to_numpy()]  # code -1 (missing) takes the last
    return area.astype(object).map(offsets).fillna(0.0).to_numpy(dtype=float)


def area_days(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per area and SGT day of `rows` (gold.model_rows): the day's recorded ticks, and whether an alert or a
    flash flood followed any of them."""
    frame = pd.DataFrame(
        {
            "area": rows["area"].astype(object).to_numpy(),
            "day": _sgt(rows["ts"]).dt.normalize().to_numpy(),
            "alert": rows["alert_label"].to_numpy(dtype=np.int8),
            "flood": rows["flood_label"].to_numpy(dtype=np.int8),
        }
    )
    return (
        frame.groupby(["area", "day"], sort=True)
        .agg(ticks=("alert", "size"), alert=("alert", "max"), flood=("flood", "max"))
        .reset_index()
    )


ALERT_CLAIMS, FLOOD_CLAIMS = "alert", "flood"


def recency(
    day: pd.Series, half_life_days: float, reference=None, *, two_sided: bool = False
) -> np.ndarray:
    """Each day's weight, 0.5 ** (days from `reference` / half_life_days): from the last day of `day` by default, days
    after the reference counting fully unless `two_sided` (module docstring)."""
    if not half_life_days > 0:
        raise ValueError(f"half_life_days is positive, not {half_life_days!r}")
    day = pd.to_datetime(pd.Series(day)).dt.normalize()
    ref = day.max() if reference is None else pd.Timestamp(reference).normalize()
    gap = (ref - day).dt.days.to_numpy(dtype=float)
    gap = np.abs(gap) if two_sided else np.clip(gap, 0.0, None)
    # never quite 0, so a day's rate (claim / exposure) stays defined
    return np.maximum(0.5 ** (gap / half_life_days), 1e-9)


def area_prior(
    days: pd.DataFrame,
    *,
    flood_weight: float = 1.0,
    claims: str = ALERT_CLAIMS,
    half_life_days: float | None = None,
    reference=None,
    two_sided: bool = False,
) -> AreaPrior:
    """The Bühlmann-Straub prior of every area in `days` (area_days rows; the module docstring has the formulas).

    `claims` is what a day claims: ALERT_CLAIMS (1 for a day with a PUB alert, `flood_weight` for one with a flash
    flood: the prior of the gated pair and of the ordinal model's alert rung) or FLOOD_CLAIMS (1 for a day with a flash
    flood: the ordinal model's flood rung). Flood days are so few that the areas' flood rates differ mostly by chance,
    and the formulas then give most of each area the island's rate. With `half_life_days`, each day's exposure and
    claim are weighed by its `recency` from `reference` (`exposure_days` is then the weighed exposure)."""
    if days.empty:
        raise ValueError("no area-days to set a prior on")
    m = days["ticks"].to_numpy(dtype=float) / TICKS_PER_DAY
    flood = days["flood"].to_numpy(dtype=float)
    if claims == FLOOD_CLAIMS:
        claim = flood
    elif claims == ALERT_CLAIMS:
        claim = days["alert"].to_numpy(dtype=float) + (flood_weight - 1.0) * flood
    else:
        raise ValueError(f"claims are {ALERT_CLAIMS!r} or {FLOOD_CLAIMS!r}, not {claims!r}")
    if half_life_days is not None:
        w = recency(days["day"], half_life_days, reference, two_sided=two_sided)
        m, claim = m * w, claim * w
    per = (
        pd.DataFrame({"area": days["area"].to_numpy(), "m": m, "c": claim})
        .groupby("area", sort=True)
        .agg(exposure_days=("m", "sum"), claims=("c", "sum"), n=("m", "size"))
    )
    rate = per["claims"] / per["exposure_days"]
    total = per["exposure_days"].sum()
    island = float(per["claims"].sum() / total)
    if island <= 0:
        raise ValueError(f"no {claims}s in the area-days to set a prior on")

    x = claim / m
    own = days["area"].map(rate).to_numpy(dtype=float)
    within = float(
        np.sum(m * (x - own) ** 2) / max(float(np.sum(per["n"] - 1)), 1.0)
    )
    spread = float(np.sum(per["exposure_days"] * (rate - island) ** 2))
    between = (spread - (len(per) - 1) * within) / (
        total - float(np.sum(per["exposure_days"] ** 2)) / total
    )
    if between > 0 and within > 0:
        k = within / between
        z = per["exposure_days"] / (per["exposure_days"] + k)
        complement = float(np.sum(z * rate) / np.sum(z))
    elif between > 0:  # no noise at all: every area is its own history
        k, z, complement = 0.0, pd.Series(1.0, index=per.index), island
    else:
        k, z, complement = float("inf"), pd.Series(0.0, index=per.index), island
    prior = (z * rate + (1 - z) * complement).clip(lower=MIN_RELATIVE_RISK * island)
    table = pd.DataFrame(
        {
            "area": per.index.astype(str),
            "exposure_days": per["exposure_days"].to_numpy(),
            "claims": per["claims"].to_numpy(),
            "rate": rate.to_numpy(),
            "credibility": z.to_numpy(),
            "prior_rate": prior.to_numpy(),
            "offset": np.log(prior.to_numpy() / island),
        }
    )
    return AreaPrior(table=table, k=float(k), island_rate=island)
