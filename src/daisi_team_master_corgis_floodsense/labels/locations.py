"""Where a PUB alert is: location text normalised for matching, PUB's sensor coordinates (SVY21) as latitude and
longitude, and the words to look a location up by.

Silver geocodes each alert location (silver.location_gazetteer) by, in order: a hand fix, PUB's own coordinates from
the flood-alerts API, the water-level sensor of the same name (the 90% warnings come from those sensors), and a OneMap
search. The alert's area is the NEA forecast area whose label point is nearest.

Alert locations are free text, spelt differently from one alert to the next and nothing like the sensors' names, so
most of this module is about deciding that two spellings name one place without a geocoder's help:

- `location_key` is the normal form: case-folded, punctuation dropped, and PUB's abbreviations spelt out ("Jln" is
  "jalan", "Bt" is "bukit"). Spelling variants of a place share a key, and silver groups, joins and caches on it.
- `name_match` scores how surely a location names a given place name, from 0 to 1: the share of words they have in
  common (leaving out words that only relate a place to a road, such as "near", "towards" and "junction", and the
  suffixes of drain sensors' names), counted against the longer of the two. SAME_NAME is the score from which two
  names are one place. Below it, matches land on the wrong stretch of a long road or on a building named after the
  road. Silver's sensor matching and ingest/onemap.py use the same bar.
- `search_queries` turns a location into OneMap queries, best guess first: as written, then without its note and
  brackets, then each road it names outside the brackets, then those inside. Queries that are too short or name no
  place ("Exit", "Service road") are dropped.
- `svy21_to_wgs84` converts the coordinates of PUB's sensor list from SVY21, Singapore's metric grid, to latitude and
  longitude, and `haversine_km` measures distances. Silver repeats the distance in Spark
  (pipeline/utilities/silver.distance_km) for its nearest-area and nearest-gauge decisions.
"""

from __future__ import annotations

import math
import re

# Abbreviations PUB uses in alert locations and sensor names, spelled out so both sides match.
_WORDS = {
    "amk": "ang mo kio",
    "ave": "avenue",
    "bt": "bukit",
    "blk": "block",
    "cck": "choa chu kang",
    "cl": "close",
    "cres": "crescent",
    "ctr": "centre",
    "center": "centre",
    "dr": "drive",
    "gdns": "gardens",
    "hts": "heights",
    "ind": "industrial",
    "jln": "jalan",
    "kg": "kampong",
    "lor": "lorong",
    "nth": "north",
    "opp": "opposite",
    "pk": "park",
    "pl": "place",
    "rd": "road",
    "sth": "south",
    "st": "street",
    "sq": "square",
    "tg": "tanjong",
    "upp": "upper",
    "c'wealth": "commonwealth",
}
# Words that say how a place relates to a road or a drain, not which place it is.
_STOP = {
    "a",
    "after",
    "along",
    "and",
    "at",
    "before",
    "between",
    "from",
    "in",
    "junction",
    "near",
    "of",
    "the",
    "to",
    "towards",
    "traffic",
    # sensor names: outlet drain, side drain, upstream, downstream, opposite
    "od",
    "sd",
    "u",
    "d",
    "s",
    "opposite",
}
_TOKEN = re.compile(r"[a-z0-9']+")


def location_key(text: str | None) -> str:
    """The same key for the same place however PUB spelt it: 'Jln Boon Lay (Bus stop B21339)' and
    'Jalan Boon Lay (Bus Stop B21339)' -> 'jalan boon lay bus stop b21339'."""
    words = (_WORDS.get(word, word) for word in _TOKEN.findall((text or "").casefold()))
    return " ".join(" ".join(words).replace("'", " ").split())


def _tokens(text: str | None) -> set[str]:
    return {word for word in location_key(text).split() if word not in _STOP}


def name_match(location: str | None, name: str | None) -> float:
    """How surely `location` names the place `name` names, from 0 to 1: the share of words they share, counted against
    the longer of the two, so 'Harper Rd' matches 'Harper Rd OD' fully but 'Bukit Timah Rd towards Clementi Rd' matches
    'Bt Timah Rd/ Cherry Ave' only partly."""
    a, b = _tokens(location), _tokens(name)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


# From this name_match score, two names name the same place. Below it, matches on the 2021-2026 archive land on the
# wrong stretch of a long road ('Upp East Coast Rd' -> an East Coast Rd sensor), or on a building named after the road
# ('Kings Rd' -> OneMap's Kings Home, Joo Chiat Road).
SAME_NAME = 0.8

_SEPARATORS = re.compile(
    r"\s*(?:/|;|,|\band\b|\btowards\b|\bnear\b|\bfrom\b|\bto\b|\bbefore\b|\bafter\b|\bbetween\b|\bat\b|\boff\b"
    r"|\bleading to\b)\s*",
    re.IGNORECASE,
)
_LEAD = re.compile(
    r"^(?:(?:traffic\s+)?junction of|intersection of|service road)\s+", re.IGNORECASE
)
_TAIL = re.compile(r"\s+(?:junction|exit|entrance)$", re.IGNORECASE)
_PREPOSITION = re.compile(
    r"^(?:after|before|near|between|towards|from|at|along|off|opp(?:osite)?)\s+",
    re.IGNORECASE,
)
# a note after a dash: "Boon Lay Way/Corporation Road Junction – 2 of 3 lanes affected"
_NOTE = re.compile(r"\s+[–—-]\s+.*$", re.DOTALL)
# words that name no place on their own, so a query of nothing else isn't worth a search
_GENERIC = {
    "exit",
    "entrance",
    "slip",
    "road",
    "service",
    "lane",
    "lanes",
    "expressway",
}


def _roads(text: str) -> list[str]:
    """The roads a phrase names: 'Junction of A and B' -> A, B; 'A towards B' -> A, B; 'after B Exit' -> B."""
    return [
        _TAIL.sub("", _LEAD.sub("", part.strip(" ,.;")))
        for part in _SEPARATORS.split(text)
    ]


def search_queries(location: str) -> list[str]:
    """What to search OneMap for, best guess first: the location as written, then without its note and brackets, then
    each road it names outside the brackets, then inside them ('KPE towards ECP (after Buangkok Drive Exit)' ends
    with 'Buangkok Drive')."""
    text = _NOTE.sub("", location.strip())
    plain = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", text)).strip(" ,.;")
    inside = [
        _PREPOSITION.sub("", part.strip()) for part in re.findall(r"\(([^)]*)\)", text)
    ]
    queries = [location.strip(), plain, *_roads(plain)]
    for part in inside:
        queries += [part, *_roads(part)]
    return [q for q in dict.fromkeys(queries) if len(q) >= 4 and _tokens(q) - _GENERIC]


# --- SVY21 (EPSG:3414), the Singapore grid PUB's sensor list uses --------------------------------------------------
# A transverse Mercator projection of the WGS84 ellipsoid (SLA): origin 1°22'N 103°50'E, false northing 38744.572 m,
# false easting 28001.642 m, scale 1. The inverse below is LINZ's series, accurate to millimetres across Singapore.

_A = 6378137.0
_F = 1 / 298.257223563
_B = _A * (1 - _F)
_E2 = 2 * _F - _F * _F
_E4, _E6 = _E2**2, _E2**3
_LAT0, _LON0 = 1 + 22 / 60, 103 + 50 / 60
_N0, _E0, _K0 = 38744.572, 28001.642, 1.0
_A0 = 1 - _E2 / 4 - 3 * _E4 / 64 - 5 * _E6 / 256
_A2 = 3 / 8 * (_E2 + _E4 / 4 + 15 * _E6 / 128)
_A4 = 15 / 256 * (_E4 + 3 * _E6 / 4)
_A6 = 35 * _E6 / 3072


def _meridian(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return _A * (
        _A0 * lat
        - _A2 * math.sin(2 * lat)
        + _A4 * math.sin(4 * lat)
        - _A6 * math.sin(6 * lat)
    )


def svy21_to_wgs84(northing: float, easting: float) -> tuple[float, float]:
    """SVY21 metres (northing Y, easting X) -> (latitude, longitude) in degrees, WGS84."""
    n = (_A - _B) / (_A + _B)
    n2, n3, n4 = n**2, n**3, n**4
    g = _A * (1 - n) * (1 - n2) * (1 + 9 * n2 / 4 + 225 * n4 / 64) * (math.pi / 180)
    m = _meridian(_LAT0) + (northing - _N0) / _K0
    sigma = m * math.pi / (180 * g)
    lat1 = (
        sigma
        + (3 * n / 2 - 27 * n3 / 32) * math.sin(2 * sigma)
        + (21 * n2 / 16 - 55 * n4 / 32) * math.sin(4 * sigma)
        + (151 * n3 / 96) * math.sin(6 * sigma)
        + (1097 * n4 / 512) * math.sin(8 * sigma)
    )
    sin2 = math.sin(lat1) ** 2
    rho = _A * (1 - _E2) / (1 - _E2 * sin2) ** 1.5
    nu = _A / math.sqrt(1 - _E2 * sin2)
    psi = nu / rho
    t = math.tan(lat1)
    t2, t4, t6 = t**2, t**4, t**6
    e = easting - _E0
    x = e / (_K0 * nu)
    x3, x5, x7 = x**3, x**5, x**7

    factor = t / (_K0 * rho)
    lat = (
        lat1
        - factor * (e * x / 2)
        + factor * (e * x3 / 24) * (-4 * psi**2 + 9 * psi * (1 - t2) + 12 * t2)
        - factor
        * (e * x5 / 720)
        * (
            8 * psi**4 * (11 - 24 * t2)
            - 12 * psi**3 * (21 - 71 * t2)
            + 15 * psi**2 * (15 - 98 * t2 + 15 * t4)
            + 180 * psi * (5 * t2 - 3 * t4)
            + 360 * t4
        )
        + factor * (e * x7 / 40320) * (1385 - 3633 * t2 + 4095 * t4 + 1575 * t6)
    )
    sec = 1 / math.cos(lat1)
    lon = (
        math.radians(_LON0)
        + x * sec
        - (x3 * sec / 6) * (psi + 2 * t2)
        + (x5 * sec / 120)
        * (
            -4 * psi**3 * (1 - 6 * t2)
            + psi**2 * (9 - 68 * t2)
            + 72 * psi * t2
            + 24 * t4
        )
        - (x7 * sec / 5040) * (61 + 662 * t2 + 1320 * t4 + 720 * t6)
    )
    return math.degrees(lat), math.degrees(lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance; at Singapore's size it is within metres of the ellipsoidal one."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))
