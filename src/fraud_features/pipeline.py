"""Command-line entry points used by the Makefile.

``python -m fraud_features.pipeline features``  generate data, build feature tables
``python -m fraud_features.pipeline train``     build the training set, train, log to MLflow
"""

from __future__ import annotations

import argparse
from datetime import datetime, time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from fraud_features.config import Settings
from fraud_features.features import (
    customer_daily_features,
    merchant_daily_features,
    txn_features,
)
from fraud_features.generator import GeneratorConfig, generate
from fraud_features.spark import get_spark
from fraud_features.storage import latest_version, read_table, write_table
from fraud_features.train import train_and_log
from fraud_features.training_set import build_training_set, leakage_violations

SOURCE_TABLES = ("transactions_raw", "labels")
FEATURE_TABLES = ("features_txn", "features_customer_daily", "features_merchant_daily")


def _date_span(transactions: DataFrame) -> tuple[str, int]:
    """First day and number of days covered, so snapshots span exactly the data."""
    row = transactions.agg(
        F.min(F.to_date("event_ts")).alias("first"), F.max(F.to_date("event_ts")).alias("last")
    ).first()
    return row["first"].isoformat(), (row["last"] - row["first"]).days + 1


def build_features(spark: SparkSession, settings: Settings, cfg: GeneratorConfig) -> None:
    transactions, labels = generate(cfg).to_frames(spark)
    for name, df in zip(SOURCE_TABLES, (transactions, labels), strict=True):
        print(f"{name}: v{write_table(df, settings.table(name))}")

    # Read back from Delta so features are built from the stored snapshot.
    transactions = read_table(spark, settings.table("transactions_raw"))
    labels = read_table(spark, settings.table("labels"))
    start, days = _date_span(transactions)
    tables = (
        txn_features(transactions),
        customer_daily_features(transactions, start, days),
        merchant_daily_features(transactions, labels, start, days),
    )
    for name, df in zip(FEATURE_TABLES, tables, strict=True):
        print(f"{name}: v{write_table(df, settings.table(name))}")


def build_and_train(
    spark: SparkSession, settings: Settings, as_of: datetime | None, maturity_days: int
) -> str:
    versions = {
        name: latest_version(spark, settings.table(name)) for name in SOURCE_TABLES + FEATURE_TABLES
    }
    frames = {name: read_table(spark, settings.table(name), v) for name, v in versions.items()}
    if as_of is None:
        last_day = frames["transactions_raw"].agg(F.max(F.to_date("event_ts"))).first()[0]
        as_of = datetime.combine(last_day, time.max).replace(microsecond=0)

    training_set = build_training_set(
        frames["transactions_raw"],
        frames["labels"],
        frames["features_txn"],
        frames["features_customer_daily"],
        frames["features_merchant_daily"],
        as_of=as_of,
        maturity_days=maturity_days,
    )
    violations = leakage_violations(training_set)
    if violations:
        raise RuntimeError(f"{violations} training rows use features from after the event")

    path = settings.table("training_set")
    version = write_table(training_set, path)
    pdf = read_table(spark, path, version).toPandas()
    lineage = {
        "training_set_path": path,
        "training_set_version": str(version),
        "as_of": as_of.isoformat(),
        "label_maturity_days": str(maturity_days),
        **{f"{name}_version": str(v) for name, v in versions.items()},
    }
    run_id = train_and_log(pdf, lineage, settings)
    print(f"training_set: v{version}, {len(pdf)} rows; MLflow run {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud feature pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    feats = sub.add_parser("features", help="generate data and build feature tables")
    feats.add_argument("--customers", type=int, default=GeneratorConfig.customers)
    feats.add_argument("--days", type=int, default=GeneratorConfig.days)
    feats.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    train = sub.add_parser("train", help="build the training set, train, log to MLflow")
    train.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    train.add_argument("--maturity-days", type=int, default=30)
    args = parser.parse_args()

    settings = Settings.from_env()
    spark = get_spark("fraud-features", settings.spark_master)
    spark.sparkContext.setLogLevel("WARN")
    if args.command == "features":
        cfg = GeneratorConfig(customers=args.customers, days=args.days, seed=args.seed)
        build_features(spark, settings, cfg)
    else:
        build_and_train(spark, settings, args.as_of, args.maturity_days)


if __name__ == "__main__":
    main()
