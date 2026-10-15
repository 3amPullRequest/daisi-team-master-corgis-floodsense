"""Silver agrees with the model contract, with bronze, and with its seed files, checked without Spark."""

import csv
import re
from pathlib import Path

from utilities.bronze_sources import CSV_TABLES, HEARTBEAT_TABLE, JSON_TABLES
from utilities.contract_tables import AREA_ALERTS_SCHEMA, AREA_ALERTS_TYPES
from utilities.forecast_categories import FORECASTS

from daisi_team_master_corgis_floodsense.labels import alert_text as T
from daisi_team_master_corgis_floodsense.ml import contract as C

ROOT = Path(__file__).resolve().parents[1]
SILVER_CODE = ROOT / "src" / "pipeline" / "transformations" / "silver"


def test_area_alerts_schema_is_the_contract():
    assert list(AREA_ALERTS_TYPES) == list(C.AREA_ALERTS_COLUMNS)
    names = [
        field.split()[0]
        for field in re.split(r",\s*(?=\w+ [A-Z]+ COMMENT)", AREA_ALERTS_SCHEMA)
    ]
    assert names == list(C.AREA_ALERTS_COLUMNS)
    assert (
        r"\'warning\' or \'flash_flood\'" in AREA_ALERTS_SCHEMA
    )  # the contract's quotes, escaped for Spark


def test_forecast_codes_agree_with_the_yearly_csvs():
    # every (text, code) pair in nea_two_hr_forecast_hist, 2021-2024 (queried 2026-10-14)
    csv_pairs = {
        "Cloudy": "CL", "Partly Cloudy (Day)": "PC", "Partly Cloudy (Night)": "PN", "Showers": "SH",
        "Light Rain": "LR", "Thundery Showers": "TL", "Windy": "WD", "Light Showers": "LS", "Fair (Night)": "FN",
        "Fair (Day)": "FA", "Moderate Rain": "RA", "Passing Showers": "PS",
        "Heavy Thundery Showers with Gusty Winds": "HG", "Heavy Thundery Showers": "HT", "Fair & Warm": "FW",
        "Heavy Showers": "HS", "Heavy Rain": "HR", "Slightly Hazy": "LH", "Mist": "BR",
    }  # fmt: skip
    assert {text: FORECASTS[text][0] for text in csv_pairs} == csv_pairs
    heavier = [
        "Cloudy",
        "Light Rain",
        "Moderate Rain",
        "Heavy Rain",
        "Thundery Showers",
        "Heavy Thundery Showers",
    ]
    assert [FORECASTS[t][1] for t in heavier] == sorted(
        FORECASTS[t][1] for t in heavier
    )


def _seed_header(name: str) -> list[str]:
    with (ROOT / "data" / "static" / f"{name}.csv").open(encoding="utf-8") as f:
        return next(csv.reader(line for line in f if not line.startswith("#")))


def test_seed_files_have_the_columns_silver_reads():
    assert _seed_header("telegram_manual_labels") == ["message_id", "category", "note"]
    assert _seed_header("location_gazetteer_manual") == [
        "location",
        "lat",
        "lon",
        "note",
    ]
    with (ROOT / "data" / "static" / "telegram_manual_labels.csv").open(
        encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    assert rows and all(row["category"] in T.CATEGORIES for row in rows)


def test_silver_reads_only_tables_that_exist():
    code = "\n".join(path.read_text("utf-8") for path in SILVER_CODE.glob("*.py"))
    read = set(re.findall(r'spark\.read(?:Stream)?\.table\("([a-z_0-9]+)"\)', code))
    defined = set(
        re.findall(r'name="([a-z_0-9]+)"', code)
    )  # temporary views and private tables
    bronze = set(JSON_TABLES) | set(CSV_TABLES) | {HEARTBEAT_TABLE}
    assert read and read <= bronze | defined, read - bronze - defined
    silver_read = set(
        re.findall(r'spark\.read\.table\(f"\{SILVER\}\.([a-z_0-9]+)"\)', code)
    )
    silver_defined = set(re.findall(r'name=f"\{SILVER\}\.([a-z_0-9]+)"', code)) | set(
        re.findall(r'TARGET = f"\{SILVER\}\.([a-z_0-9]+)"', code)
    )
    assert silver_read <= silver_defined, silver_read - silver_defined
