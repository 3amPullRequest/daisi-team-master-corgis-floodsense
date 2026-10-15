"""NEA 2-hour forecast categories: NEA's code for each, and how much rain it promises. Pure data, so it imports
without Spark.

The yearly CSVs carry the code; the v2 API sends only the text. Codes are as the 2021-2024 CSVs pair them, plus the
few categories they never used. `rain_class` orders the categories by the rain they forecast, so a model can read it as
a number: 0 none, 1 light, 2 moderate, 3 heavy, 4 thundery, 5 heavy thundery.
"""

FORECASTS: dict[str, tuple[str, int]] = {
    "Fair": ("FA", 0),
    "Fair (Day)": ("FA", 0),
    "Fair (Night)": ("FN", 0),
    "Fair and Warm": ("FW", 0),
    "Fair & Warm": ("FW", 0),
    "Partly Cloudy": ("PC", 0),
    "Partly Cloudy (Day)": ("PC", 0),
    "Partly Cloudy (Night)": ("PN", 0),
    "Cloudy": ("CL", 0),
    "Hazy": ("HZ", 0),
    "Slightly Hazy": ("LH", 0),
    "Windy": ("WD", 0),
    "Mist": ("BR", 0),
    "Fog": ("FG", 0),
    "Light Rain": ("LR", 1),
    "Light Showers": ("LS", 1),
    "Passing Showers": ("PS", 1),
    "Moderate Rain": ("RA", 2),
    "Showers": ("SH", 2),
    "Heavy Rain": ("HR", 3),
    "Heavy Showers": ("HS", 3),
    "Thundery Showers": ("TL", 4),
    "Heavy Thundery Showers": ("HT", 5),
    "Heavy Thundery Showers with Gusty Winds": ("HG", 5),
}
