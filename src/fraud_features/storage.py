"""Versioned Delta reads and writes.

Every write returns the Delta version it produced. That number is what goes
into MLflow, so a training run can always be traced to, and rebuilt from, the
exact snapshot of each table it used (``versionAsOf``).
"""

from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession


def write_table(df: DataFrame, path: str) -> int:
    """Overwrite ``path`` as a new Delta version and return that version."""
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    return latest_version(df.sparkSession, path)


def latest_version(spark: SparkSession, path: str) -> int:
    return DeltaTable.forPath(spark, path).history(1).collect()[0]["version"]


def read_table(spark: SparkSession, path: str, version: int | None = None) -> DataFrame:
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    return reader.load(path)
