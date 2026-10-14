"""The ingestion scripts and the bronze pipeline agree on folder names and formats."""

from utilities.bronze_sources import CSV_TABLES, HEARTBEAT_TABLE, JSON_TABLES

from daisi_team_master_corgis_floodsense.ingest import sources as S


def test_every_landing_folder_has_a_bronze_table_and_back():
    assert (
        set(JSON_TABLES) | set(CSV_TABLES) | {HEARTBEAT_TABLE}
        == S.bronze_landing_names()
    )


def test_csv_tables_are_exactly_the_folders_written_as_csv():
    written_as_csv = {s.hist_name for s in S.API_SOURCES if s.csv_collection}
    written_as_csv |= {layer.name for layer in S.STATIC_LAYERS if layer.kind == "csv"}
    assert set(CSV_TABLES) == written_as_csv | {S.UHSLC}
