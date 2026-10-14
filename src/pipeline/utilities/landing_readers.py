"""Auto Loader readers behind the bronze tables: files stream in as they land, unchanged."""

from pyspark import pipelines as dp
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _file_columns() -> list:
    return [
        F.col("_metadata.file_path").alias("source_file"),
        F.col("_metadata.file_modification_time").alias("file_modified_at"),
        F.current_timestamp().alias("ingested_at"),
    ]


def _autoloader(spark: SparkSession, fmt: str):
    # Landing folders have date=/year=/version= levels; they stay in source_file instead of becoming columns.
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", fmt)
        .option("cloudFiles.partitionColumns", "")
    )


def json_lines(spark: SparkSession, path: str) -> DataFrame:
    """One row per line of every *.jsonl / *.jsonl.gz file, the whole line parsed into a VARIANT `payload`."""
    lines = _autoloader(spark, "text").option("pathGlobFilter", "*.jsonl*").load(path)
    parsed = lines.select(
        F.expr("try_parse_json(value)").alias("payload"), *_file_columns()
    )
    fetched_at = F.expr("try_variant_get(payload, '$.fetched_at', 'string')")
    return parsed.select(
        "payload",
        F.to_timestamp(fetched_at).alias("fetched_at"),
        "source_file",
        "file_modified_at",
        "ingested_at",
    )


def csv_files(
    spark: SparkSession, path: str, *, schema: str, header: bool = True
) -> DataFrame:
    """Every *.csv / *.csv.gz file with the given all-text schema; unexpected columns land in _rescued_data."""
    reader = (
        _autoloader(spark, "csv")
        .schema(schema)
        .option("header", str(header).lower())
        .option("rescuedDataColumn", "_rescued_data")
        .option("pathGlobFilter", "*.csv*")
    )
    return reader.load(path).select("*", *_file_columns())


def json_table(spark: SparkSession, landing: str, name: str, comment: str) -> None:
    @dp.table(name=name, comment=comment)
    @dp.expect("payload_parsed", "payload IS NOT NULL")
    def bronze():
        return json_lines(spark, f"{landing}/{name}")


def csv_table(
    spark: SparkSession,
    landing: str,
    name: str,
    *,
    comment: str,
    schema: str,
    header: bool = True,
) -> None:
    @dp.table(name=name, comment=comment)
    def bronze():
        return csv_files(spark, f"{landing}/{name}", schema=schema, header=header)
