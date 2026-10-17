"""Which bronze tables the weather and land-use silver tables read, and what they keep. Pure data, so it imports
without Spark.

Two small catalogues, each shared by more than one silver or gold table:

- WEATHER_VARIABLES names NEA's four 1-minute variables with their units (air_temp_c, rel_humidity_pct, wind_speed_kn,
  wind_dir_deg), the bronze API table each comes from (its yearly-CSV twin is `<table>_hist`), and the range of values
  silver keeps. Readings outside the range are dropped as errors. The ranges are wide physical limits around
  WEATHER_EXTREMES_2021_2024, the real extremes of the 110 million CSV readings, and a test checks that the extremes
  fall inside them.
- MASTER_PLANS names the two URA Master Plan land-use tables with the plan's name and the days it is in force. MP2019
  runs until MP2025 is gazetted on 1 Dec 2025, so a tick reads the plan of its day (silver.land_use).
"""

from datetime import date

# NEA's 1-minute weather readings: variable (named with its unit) -> (bronze API table, (lowest, highest) value kept).
# Each API table has a yearly-CSV twin, `<table>_hist`. The bounds are wide physical limits around the extremes of the
# 2021-2024 CSVs (WEATHER_EXTREMES_2021_2024); silver drops a reading outside them.
WEATHER_VARIABLES: dict[str, tuple[str, tuple[float, float]]] = {
    "air_temp_c": ("nea_air_temperature", (15.0, 40.0)),
    "rel_humidity_pct": ("nea_relative_humidity", (0.0, 100.0)),
    "wind_speed_kn": ("nea_wind_speed", (0.0, 60.0)),
    "wind_dir_deg": ("nea_wind_direction", (0.0, 360.0)),
}

# The smallest and largest value of each variable in the 2021-2024 CSVs (110 million readings, queried 2026-10-15).
# Wind speed's 49.4 knots is Sentosa on 15-16 Mar 2024; only 156 readings in four years exceed 30.
WEATHER_EXTREMES_2021_2024: dict[str, tuple[float, float]] = {
    "air_temp_c": (21.0, 36.0),
    "rel_humidity_pct": (30.0, 100.0),
    "wind_speed_kn": (0.0, 49.4),
    "wind_dir_deg": (0.0, 359.0),
}

# URA Master Plan land-use layers: bronze table -> (plan, first day in force, first day no longer in force). MP2019
# was approved on 18 Nov 2019 and MP2025 gazetted on 1 Dec 2025, so a tick before 1 Dec 2025 reads MP2019.
MASTER_PLANS: dict[str, tuple[str, date, date | None]] = {
    "static_mp2019_land_use": ("MP2019", date(2019, 11, 18), date(2025, 12, 1)),
    "static_mp2025_land_use": ("MP2025", date(2025, 12, 1), None),
}
