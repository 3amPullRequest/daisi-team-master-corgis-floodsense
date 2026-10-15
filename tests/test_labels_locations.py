import pytest

from daisi_team_master_corgis_floodsense.labels.locations import (
    haversine_km,
    location_key,
    name_match,
    search_queries,
    svy21_to_wgs84,
)


def test_location_key_is_the_same_however_pub_spelt_it():
    assert location_key("Jln Boon Lay (Bus stop B21339)") == location_key(
        "Jalan Boon Lay (Bus Stop B21339)"
    )
    assert (
        location_key("Upp Paya Lebar Rd")
        == location_key("Upper Paya Lebar Road")
        == "upper paya lebar road"
    )
    assert location_key("Alexandra Canal (C'wealth Ave u/s culvert)").startswith(
        "alexandra canal commonwealth avenue"
    )
    assert location_key(None) == location_key("  ") == ""


def test_name_match_counts_shared_words_against_the_longer_name():
    assert name_match("Boon Lay Way (Blk 201)", "Boon Lay Way (Blk 201)") == 1.0
    assert (
        name_match("Harper Rd", "Harper Rd OD") == 1.0
    )  # "OD" (outlet drain) says nothing about the place
    assert (
        name_match("Puay Hee Ave / Siak Kew Ave", "Puay Hee Ave/ Siak Kew Ave") == 1.0
    )
    # a long road shared with another sensor is not enough
    assert (
        name_match("Bukit Timah Rd towards Clementi Rd", "Bt Timah Rd/ Cherry Ave")
        < 0.8
    )
    assert name_match("Upp East Coast Rd", "Tembeling Rd OD (East Coast Rd)") < 0.8
    assert name_match("", "Harper Rd") == 0.0


def test_search_queries_go_from_the_full_text_to_each_road():
    assert search_queries("TPE Slip Rd (Punggol West Flyover)") == [
        "TPE Slip Rd (Punggol West Flyover)",
        "TPE Slip Rd",
        "Punggol West Flyover",
    ]
    assert search_queries("Junction of Stevens Road and Balmoral Road") == [
        "Junction of Stevens Road and Balmoral Road",
        "Stevens Road",
        "Balmoral Road",
    ]
    assert "Siak Kew Ave" in search_queries("Puay Hee Ave / Siak Kew Ave")


def test_svy21_to_wgs84_matches_onemap():
    # OneMap's answer for "Boon Lay Way" gives both: X 16216.62348679081, Y 36277.26471230201
    lat, lon = svy21_to_wgs84(36277.26471230201, 16216.62348679081)
    assert (
        haversine_km(lat, lon, 1.344350907310431, 103.7274378185247) < 1e-6
    )  # under a millimetre
    # the grid's origin is 1°22'N 103°50'E
    assert svy21_to_wgs84(38744.572, 28001.642) == pytest.approx(
        (1 + 22 / 60, 103 + 50 / 60), abs=1e-9
    )


def test_haversine_km():
    # one degree of longitude at the equator, on the mean sphere
    assert haversine_km(0, 103, 0, 104) == pytest.approx(111.195, abs=0.01)


def test_search_queries_split_brackets_and_drop_notes():
    # the bracket's roads, without the preposition; generic words alone aren't searched
    assert (
        search_queries("KPE towards ECP (after Buangkok Drive Exit)")[-1]
        == "Buangkok Drive"
    )
    assert "Slip Rd" not in search_queries("KPE (Slip Rd to Buangkok East Drive)")
    assert search_queries("Kings Rd (between Prince Rd and Lutheran Rd)")[-2:] == [
        "Prince Rd",
        "Lutheran Rd",
    ]
    # a note after a dash goes
    assert search_queries(
        "Boon Lay Way/Corporation Road Junction – 2 of 3 lanes affected"
    )[1:] == [
        "Boon Lay Way/Corporation Road Junction",
        "Boon Lay Way",
        "Corporation Road",
    ]
    assert "exit" not in search_queries(
        "PIE slip road entrance / exit at Bedok North Ave 3 exit"
    )
