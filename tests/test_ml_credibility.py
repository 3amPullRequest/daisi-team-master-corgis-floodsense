import math

import numpy as np
import pandas as pd
import pytest

from daisi_team_master_corgis_floodsense.ml import credibility as K

DAY = K.TICKS_PER_DAY


def days(claims: dict[str, list[int]], floods: dict[str, list[int]] | None = None, ticks=DAY) -> pd.DataFrame:
    """area_days rows: per area, one full day per claim (1 an alert day, 0 a quiet one)."""
    rows = []
    for area, alerts in claims.items():
        for d, alert in enumerate(alerts):
            flood = (floods or {}).get(area, [0] * len(alerts))[d]
            rows.append((area, pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), ticks, alert, flood))
    return pd.DataFrame(rows, columns=["area", "day", "ticks", "alert", "flood"])


def test_buhlmann_straub_by_hand():
    # Three areas, four full days each: alert days 2, 0 and 3. By hand: process variance s2 = 1.75 / 9, the areas'
    # variance v = (7/6 - 2 s2) / 8 = 7/72, so k = s2 / v = 2 days and every Z = 4 / (4 + 2) = 2/3. The island's rate
    # is 5/12 and the Z-weighted mean of the rates is too, so the area without an alert keeps a third of it.
    prior = K.area_prior(days({"A": [1, 0, 1, 0], "B": [0, 0, 0, 0], "C": [1, 1, 1, 0]}))
    t = prior.table.set_index("area")
    assert prior.k == pytest.approx(2.0)
    assert t["credibility"].tolist() == pytest.approx([2 / 3] * 3)
    assert prior.island_rate == pytest.approx(5 / 12)
    expected = {"A": 2 / 3 * 0.5 + 5 / 36, "B": 5 / 36, "C": 2 / 3 * 0.75 + 5 / 36}
    for area, rate in expected.items():
        assert t.loc[area, "prior_rate"] == pytest.approx(rate)
        assert t.loc[area, "offset"] == pytest.approx(math.log(rate / (5 / 12)))
    assert t.loc["B", "offset"] == pytest.approx(math.log(1 / 3))


def test_recency_halves_a_day_s_weight_every_half_life():
    day = pd.Series(pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"]))
    w = K.recency(day, 366)  # from the last day: 2024 was a leap year
    assert w.tolist() == pytest.approx([0.5, 0.5 ** (184 / 366), 1.0])
    # two-sided from a reference in the middle; one-sided, later days count fully
    mid = pd.Timestamp("2024-07-01")
    assert K.recency(day, 182, mid, two_sided=True).tolist() == pytest.approx([0.5, 1.0, 0.5 ** (184 / 182)])
    assert K.recency(day, 182, mid).tolist() == pytest.approx([0.5, 1.0, 1.0])
    with pytest.raises(ValueError):
        K.recency(day, 0)


def test_a_half_life_trusts_an_area_s_recent_years_more():
    # A alerted only in its first 40 days, B only in its last 40, on the same 120 days
    early, late = [1] * 40 + [0] * 80, [0] * 80 + [1] * 40
    flat = K.area_prior(days({"A": early, "B": late, "C": [0, 1, 0] * 40})).table.set_index("area")
    recent = K.area_prior(days({"A": early, "B": late, "C": [0, 1, 0] * 40}), half_life_days=20).table.set_index("area")
    # alike without recency (the same 40 alert days each); with it, B's recent alerts count for far more than A's old ones
    assert flat.loc["A", "prior_rate"] == pytest.approx(flat.loc["B", "prior_rate"])
    assert recent.loc["B", "prior_rate"] > recent.loc["A", "prior_rate"]
    assert np.isfinite(recent["offset"]).all()


def test_an_area_without_alerts_gets_a_finite_prior_below_the_others():
    claims = {"wet": [1, 0, 1, 1, 0, 1] * 5, "dry": [0] * 30, "mid": [0, 0, 1, 0, 0, 0] * 5}
    t = K.area_prior(days(claims)).table.set_index("area")
    assert np.isfinite(t["offset"]).all()
    assert t.loc["dry", "offset"] < t.loc["mid", "offset"] < t.loc["wet", "offset"]
    # shrunk towards the island, never past it
    assert (t["prior_rate"] > 0).all()
    assert t.loc["dry", "prior_rate"] > t.loc["dry", "rate"] == 0
    assert t.loc["wet", "prior_rate"] < t.loc["wet", "rate"]


def test_the_more_an_area_has_been_seen_the_more_its_own_history_counts():
    short = days({"A": [1, 0] * 3, "B": [0] * 6, "C": [0, 1, 0] * 2})
    long = days({"A": [1, 0] * 30, "B": [0] * 60, "C": [0, 1, 0] * 20})
    z_short = K.area_prior(short).table["credibility"].mean()
    z_long = K.area_prior(long).table["credibility"].mean()
    assert 0 < z_short < z_long < 1


def test_areas_that_differ_only_by_chance_all_get_the_island():
    # the same history in every area: no spread beyond noise, so no area's own history counts
    prior = K.area_prior(days({"A": [1, 0, 0, 1], "B": [1, 0, 0, 1], "C": [1, 0, 0, 1]}))
    assert math.isinf(prior.k)
    assert prior.table["credibility"].tolist() == [0.0, 0.0, 0.0]
    assert prior.table["offset"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_a_flood_day_weighs_flood_weight():
    claims = {"A": [1, 0, 0, 0] * 5, "B": [1, 0, 0, 0] * 5, "C": [0] * 20}
    floods = {"A": [1, 0, 0, 0] * 5, "B": [0] * 20, "C": [0] * 20}
    plain = K.area_prior(days(claims, floods)).table.set_index("area")
    weighted = K.area_prior(days(claims, floods), flood_weight=10).table.set_index("area")
    assert plain.loc["A", "claims"] == plain.loc["B", "claims"] == 5
    assert weighted.loc["A", "claims"] == 50 and weighted.loc["B", "claims"] == 5
    assert weighted.loc["A", "offset"] > weighted.loc["B", "offset"]
    assert plain.loc["A", "offset"] == pytest.approx(plain.loc["B", "offset"])


def test_exposure_is_the_recorded_share_of_each_day():
    half = days({"A": [1, 0], "B": [0, 0]}, ticks=DAY // 2)
    t = K.area_prior(half).table.set_index("area")
    assert t["exposure_days"].tolist() == [1.0, 1.0]
    assert t.loc["A", "rate"] == 1.0  # one alert day over one recorded day


def test_area_days_from_model_rows():
    ts = pd.date_range("2024-01-01", periods=3 * DAY, freq="5min")
    rows = pd.DataFrame(
        {
            "area": pd.Categorical(["A"] * len(ts), categories=["A", "B"]),
            "ts": ts,
            "alert_label": np.r_[np.zeros(DAY + 10), np.ones(3), np.zeros(2 * DAY - 13)].astype(int),
            "flood_label": 0,
        }
    )
    d = K.area_days(rows)
    assert d["day"].tolist() == list(pd.date_range("2024-01-01", periods=3, freq="D"))
    assert d["alert"].tolist() == [0, 1, 0] and (d["ticks"] == DAY).all()


def test_offsets_of_an_unknown_area_are_the_islands():
    area = pd.Series(pd.Categorical(["A", "B", "Z", None], categories=["A", "B", "Z"]))
    got = K.area_offsets(area, {"A": -1.5, "B": 0.5})
    assert got.tolist() == [-1.5, 0.5, 0.0, 0.0]
    assert K.area_offsets(area.astype(object), {"A": -1.5}).tolist() == [-1.5, 0.0, 0.0, 0.0]


def test_no_alerts_no_prior():
    with pytest.raises(ValueError, match="no alerts"):
        K.area_prior(days({"A": [0, 0], "B": [0, 0]}))


def test_the_flood_rung_prior_counts_flood_days_and_shrinks_them_to_the_island():
    # two flood days among three areas' alert days: the areas' flood rates differ by chance, so all get the island's
    d = days({"A": [1, 1, 0, 0], "B": [1, 0, 0, 0], "C": [0, 0, 0, 0]}, floods={"A": [1, 0, 0, 0], "B": [1, 0, 0, 0]})
    flood = K.area_prior(d, claims=K.FLOOD_CLAIMS)
    assert flood.island_rate == pytest.approx(2 / 12)
    alert = K.area_prior(d)
    assert alert.island_rate == pytest.approx(3 / 12)
    # an area that never flooded keeps most of the island's flood rate
    assert flood.table.set_index("area").loc["C", "prior_rate"] >= 0.5 * flood.island_rate
    with pytest.raises(ValueError, match="claims are"):
        K.area_prior(d, claims="warning")
    with pytest.raises(ValueError, match="no floods"):
        K.area_prior(days({"A": [1, 0]}), claims=K.FLOOD_CLAIMS)
