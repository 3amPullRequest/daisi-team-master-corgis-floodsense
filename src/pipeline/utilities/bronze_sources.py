"""Bronze tables and their landing folders: `bronze.<name>` streams every file under `<landing_path>/<name>/`.

The ingestion scripts (src/ingestion) write those folders; tests/test_bronze_contract.py keeps both sides in step.
Pure data, so it imports without Spark.
"""

# JSON Lines folders: every line becomes one row with the whole line as `payload` (VARIANT).
JSON_TABLES: dict[str, str] = {
    "nea_rainfall": "NEA rain gauges, 5-minute totals: raw v2 API pages (live ticks and day backfill)",
    "nea_air_temperature": "NEA air temperature, 1-minute: raw v2 API pages",
    "nea_relative_humidity": "NEA relative humidity, 1-minute: raw v2 API pages",
    "nea_wind_speed": "NEA wind speed, 1-minute: raw v2 API pages",
    "nea_wind_direction": "NEA wind direction, 1-minute: raw v2 API pages",
    "nea_two_hr_forecast": "NEA 2-hour forecast for the 47 areas: raw v2 API pages",
    "nea_lightning": "NEA lightning observations: raw v2 API pages",
    "nea_radar_70km": "NEA 70 km rain radar metadata: raw v2 API pages (frames in landing/radar_png)",
    "pub_flood_alerts": "PUB flash-flood alerts (CAP, with lat/lon circles): raw v2 API pages",
    "pub_telegram_posts": "PUB Flood Alerts Telegram posts: 90% drain warnings, flash floods, NEA heavy-rain warnings (live, channel scrape, and posts PUB deleted, from web archive captures)",
    "nea_tide_timings": "NEA tide table rows (time, height, high/low), scraped daily",
    "onemap_search": "OneMap search answers for PUB alert locations, one line per query (raw answer in body), for "
    "silver's gazetteer",
    "static_mp2019_land_use": "URA Master Plan 2019 land use, one GeoJSON feature per row",
    "static_mp2025_land_use": "URA Master Plan 2025 land use, one GeoJSON feature per row",
    "static_pub_water_level_sensors": "PUB drain water-level sensors (SVY21 X/Y), one spreadsheet row per row",
    "static_sla_vertical_control": "SLA vertical control benchmarks, one GeoJSON feature per row",
}

HEARTBEAT_TABLE = "ingest_heartbeat"
HEARTBEAT_COMMENT = (
    "One row per ingestion tick: per-source status, record counts and latency"
)


def text_columns(*names: str) -> str:
    return ", ".join(f"{name} STRING" for name in names)


# data.gov.sg yearly CSV headers (the same in the 2021 and 2024 files; relative humidity has no reading_type)
_READING_COLUMNS = (
    "date",
    "timestamp",
    "update_timestamp",
    "station_id",
    "station_name",
    "station_device_id",
    "location_longitude",
    "location_latitude",
    "reading_update_timestamp",
    "reading_value",
    "reading_type",
    "reading_unit",
)
_READINGS = text_columns(*_READING_COLUMNS)

# CSV folders, every column as text. The schema is explicit, so nothing is inferred and a folder holding only a
# half-written file is fine; Auto Loader matches columns by header and rescues any it doesn't know.
CSV_TABLES: dict[str, dict] = {
    "nea_rainfall_hist": {
        "comment": "data.gov.sg Historical Rainfall across Singapore, yearly CSVs",
        "schema": _READINGS,
    },
    "nea_two_hr_forecast_hist": {
        "comment": "data.gov.sg Historical 2-hour Weather Forecast, yearly CSVs",
        "schema": text_columns(
            "date",
            "timestamp",
            "valid_period_text",
            "valid_period_start",
            "valid_period_end",
            "update_timestamp",
            "location_name",
            "location_longitude",
            "location_latitude",
            "forecast_text",
            "forecast_code",
        ),
    },
    "nea_air_temperature_hist": {
        "comment": "data.gov.sg Historical Air Temperature across Singapore, yearly CSVs",
        "schema": _READINGS,
    },
    "nea_relative_humidity_hist": {
        "comment": "data.gov.sg Historical Relative Humidity across Singapore, yearly CSVs",
        "schema": text_columns(*(c for c in _READING_COLUMNS if c != "reading_type")),
    },
    "nea_wind_speed_hist": {
        "comment": "data.gov.sg Historical Wind Speed across Singapore, yearly CSVs",
        "schema": _READINGS,
    },
    "nea_wind_direction_hist": {
        "comment": "data.gov.sg Historical Wind Direction across Singapore, yearly CSVs",
        "schema": _READINGS,
    },
    "uhslc_tanjong_pagar_hourly": {
        "comment": "UHSLC research-quality hourly sea level at Tanjong Pagar (mm; -32767 = missing)",
        "header": False,
        "schema": text_columns("year", "month", "day", "hour", "sea_level_mm"),
    },
    "static_flood_prone_area_ha": {
        "comment": "PUB flood-prone area in hectares, by year",
        "schema": text_columns("Year", "Hectares"),
    },
}
