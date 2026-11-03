"""Assemble a training set as it could have been built on a given day.

Two clocks matter. ``as_of`` is the day we pretend to be training: labels
reported after it do not exist yet. ``maturity_days`` excludes transactions too
recent for their chargebacks to have arrived; without it, recent fraud would
be labelled "legit" simply because nobody has reported it yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from fraud_features.features import FEATURE_COLUMNS
from fraud_features.pit import point_in_time_join

FEATURE_TS_COLUMNS = ("txn_feature_ts", "customer_feature_ts", "merchant_feature_ts")


def ts_literal(value: datetime) -> Column:
    """A timestamp literal read in the session time zone (UTC), not the host's."""
    return F.lit(value.strftime("%Y-%m-%d %H:%M:%S")).cast("timestamp")


def labels_as_of(
    transactions: DataFrame, labels: DataFrame, as_of: datetime, maturity_days: int
) -> DataFrame:
    """Mature transactions with the label that was known at ``as_of`` (0 if none)."""
    cutoff = as_of - timedelta(days=maturity_days)
    known = labels.where(F.col("label_ts") <= ts_literal(as_of)).select("txn_id", "is_fraud")
    return (
        transactions.where(F.col("event_ts") <= ts_literal(cutoff))
        .join(known, "txn_id", "left")
        .select(
            "txn_id",
            "customer_id",
            "merchant_id",
            "event_ts",
            F.coalesce("is_fraud", F.lit(0)).alias("is_fraud"),
        )
    )


def build_training_set(
    transactions: DataFrame,
    labels: DataFrame,
    txn_features: DataFrame,
    customer_daily: DataFrame,
    merchant_daily: DataFrame,
    as_of: datetime,
    maturity_days: int,
) -> DataFrame:
    base = labels_as_of(transactions, labels, as_of, maturity_days)
    with_txn = base.join(
        txn_features.drop("customer_id", "merchant_id", "event_ts").withColumnRenamed(
            "feature_ts", "txn_feature_ts"
        ),
        "txn_id",
        "left",
    )
    with_customer = point_in_time_join(
        with_txn, customer_daily, ["customer_id"], "event_ts", "txn_id", "customer_feature_ts"
    )
    full = point_in_time_join(
        with_customer, merchant_daily, ["merchant_id"], "event_ts", "txn_id", "merchant_feature_ts"
    )
    return full.select(
        "txn_id",
        "customer_id",
        "merchant_id",
        "event_ts",
        *FEATURE_TS_COLUMNS,
        *FEATURE_COLUMNS,
        "is_fraud",
    )


def leakage_violations(training_set: DataFrame) -> int:
    """Rows where any feature was computed after the event. Must be zero.

    The join logic already guarantees this; checking the output as well turns
    a silent modelling bug into a failed pipeline run if someone changes it.
    """
    future = [F.col(c) > F.col("event_ts") for c in FEATURE_TS_COLUMNS]
    condition = future[0]
    for check in future[1:]:
        condition = condition | check
    return training_set.where(F.coalesce(condition, F.lit(False))).count()
