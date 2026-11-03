"""Local SparkSession with Delta, version-matched to the ``delta-spark`` wheel."""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def get_spark(app_name: str, master: str = "local[2]") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Feature timestamps must mean the same thing on every machine.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()
