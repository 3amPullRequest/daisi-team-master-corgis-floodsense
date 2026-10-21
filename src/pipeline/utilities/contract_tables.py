"""Spark schemas for the tables ml/contract.py defines, with the contract's descriptions as column comments. Pure data
(no Spark), so tests/test_silver_contract.py can check it against the contract.

The contract says what each column of a shared table means and in which order; this file adds the Spark type of each.
`ddl` joins the two into the `name TYPE COMMENT '...'` string a table is declared with (`schema=` of the pipeline's
materialized views), so a table is created with exactly the columns, order and descriptions the contract gives, and the
description a reader sees on a column is the one the models rely on. A column in the contract without a type here fails
at import, so the two can't drift apart unnoticed. The model rows reuse the area-features types for the features they
share.
"""

from daisi_team_master_corgis_floodsense.ml import contract as C

AREA_ALERTS_TYPES = {
    "area": "STRING",
    "alert_ts": "TIMESTAMP",
    "kind": "STRING",
    "posted_ts": "TIMESTAMP",
    "location": "STRING",
    "source": "STRING",
}


def ddl(columns: dict[str, str], types: dict[str, str]) -> str:
    """'name TYPE COMMENT '...', ...' in the contract's column order."""

    def quoted(text: str) -> str:
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    return ", ".join(
        f"{name} {types[name]} COMMENT {quoted(doc)}" for name, doc in columns.items()
    )


AREA_ALERTS_SCHEMA = ddl(C.AREA_ALERTS_COLUMNS, AREA_ALERTS_TYPES)

AREA_FEATURES_TYPES = {
    "area": "STRING",
    "ts": "TIMESTAMP",
    **{C.rain_column(w): "DOUBLE" for w in C.RAIN_WINDOWS_MINUTES},
    "rain_gauges": "INT",
    "forecast_rain_class": "INT",
    "forecast_age_min": "DOUBLE",
    "rain_nearby_30min_mm": "DOUBLE",
    "rain_upwind_30min_mm": "DOUBLE",
    "air_temp_c": "DOUBLE",
    "rel_humidity_pct": "DOUBLE",
    "wind_u_kn": "DOUBLE",
    "wind_v_kn": "DOUBLE",
    **{C.lightning_column(km): "INT" for km in C.LIGHTNING_RADII_KM},
    "tide_height_m": "DOUBLE",
    "minutes_to_high_tide": "DOUBLE",
}
AREA_FEATURES_SCHEMA = ddl(C.AREA_FEATURES_COLUMNS, AREA_FEATURES_TYPES)

FEATURE_TICKS_TYPES = {
    "ts": "TIMESTAMP",
    "rain_stamps": "INT",
    "forecast_areas": "INT",
    "feature_window": "STRING",
    "missing_reason": "STRING",
}
FEATURE_TICKS_SCHEMA = ddl(C.FEATURE_TICKS_COLUMNS, FEATURE_TICKS_TYPES)

MODEL_ROWS_TYPES = {
    **{c: AREA_FEATURES_TYPES[c] for c in ("area", "ts", *C.MODEL_FEATURES)},
    "alert_label": "INT",
    "flood_label": "INT",
    "split": "STRING",
}
MODEL_ROWS_SCHEMA = ddl(C.MODEL_ROWS_COLUMNS, MODEL_ROWS_TYPES)

LABEL_TICKS_TYPES = {
    "ts": "TIMESTAMP",
    "label_window": "STRING",
    "missing_reason": "STRING",
    "split": "STRING",
}
LABEL_TICKS_SCHEMA = ddl(C.LABEL_TICKS_COLUMNS, LABEL_TICKS_TYPES)

AREA_LABELS_TYPES = {
    "area": "STRING",
    "ts": "TIMESTAMP",
    "alert_status": "STRING",
    "alert_label": "INT",
    "flood_label": "INT",
    "missing_reason": "STRING",
    "split": "STRING",
}
AREA_LABELS_SCHEMA = ddl(C.AREA_LABELS_COLUMNS, AREA_LABELS_TYPES)
