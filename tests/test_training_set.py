from datetime import datetime, timedelta

import pytest

from fraud_features.features import (
    customer_daily_features,
    merchant_daily_features,
    txn_features,
)
from fraud_features.generator import LABEL_SCHEMA, TRANSACTION_SCHEMA, GeneratorConfig, generate
from fraud_features.training_set import build_training_set, labels_as_of, leakage_violations

D = lambda day: datetime(2026, 1, day, 12)  # noqa: E731


def test_labels_reported_after_as_of_are_not_known(spark):
    transactions = spark.createDataFrame(
        [
            ("old_fraud", "c1", "m1", "grocery", "online", 10.0, D(1)),
            ("late_report", "c1", "m1", "grocery", "online", 10.0, D(2)),
            ("too_recent", "c1", "m1", "grocery", "online", 10.0, D(25)),
        ],
        TRANSACTION_SCHEMA,
    )
    labels = spark.createDataFrame(
        [
            ("old_fraud", 1, D(5)),
            ("late_report", 1, D(31)),  # chargeback after the as-of date
            ("too_recent", 1, D(28)),
        ],
        LABEL_SCHEMA,
    )

    rows = labels_as_of(transactions, labels, as_of=D(30), maturity_days=7).collect()

    assert {r.txn_id: r.is_fraud for r in rows} == {"old_fraud": 1, "late_report": 0}


@pytest.fixture(scope="module")
def synthetic(spark):
    cfg = GeneratorConfig(customers=150, merchants=40, days=40, seed=5)
    transactions, labels = generate(cfg).to_frames(spark)
    start = cfg.start.date().isoformat()
    return {
        "transactions": transactions.cache(),
        "labels": labels.cache(),
        "txn_features": txn_features(transactions).cache(),
        "customer_daily": customer_daily_features(transactions, start, cfg.days).cache(),
        "merchant_daily": merchant_daily_features(transactions, labels, start, cfg.days).cache(),
        "as_of": datetime(2026, 2, 9),
    }


def _build(s, **overrides):
    args = {**s, **overrides}
    return build_training_set(
        args["transactions"],
        args["labels"],
        args["txn_features"],
        args["customer_daily"],
        args["merchant_daily"],
        as_of=args["as_of"],
        maturity_days=10,
    )


def test_training_set_has_no_future_features(synthetic):
    training_set = _build(synthetic)
    assert training_set.count() > 0
    assert leakage_violations(training_set) == 0


def test_leakage_check_catches_a_naive_join(synthetic, spark):
    # Joining the *last* snapshot per customer, ignoring time, is the classic bug.
    from pyspark.sql import functions as F

    latest = (
        synthetic["customer_daily"]
        .groupBy("customer_id")
        .agg(F.max("feature_ts").alias("feature_ts"))
    )
    naive = (
        _build(synthetic)
        .drop("customer_feature_ts")
        .join(latest.withColumnRenamed("feature_ts", "customer_feature_ts"), "customer_id")
    )
    assert leakage_violations(naive) > 0


def test_adding_future_data_does_not_change_the_training_set(synthetic, spark):
    cfg = GeneratorConfig(customers=150, merchants=40, days=40, seed=5)
    future = synthetic["as_of"] + timedelta(days=1)
    extra_txn = spark.createDataFrame(
        [("T_future", "C00001", "M0001", "grocery", "online", 999.0, future)], TRANSACTION_SCHEMA
    )
    # A chargeback for an old transaction, reported after as_of.
    old_txn = synthetic["transactions"].orderBy("event_ts").first().txn_id
    extra_label = spark.createDataFrame([(old_txn, 1, future)], LABEL_SCHEMA)

    transactions = synthetic["transactions"].union(extra_txn)
    labels = synthetic["labels"].union(extra_label)
    start = cfg.start.date().isoformat()
    with_future = _build(
        synthetic,
        transactions=transactions,
        labels=labels,
        txn_features=txn_features(transactions),
        customer_daily=customer_daily_features(transactions, start, cfg.days + 2),
        merchant_daily=merchant_daily_features(transactions, labels, start, cfg.days + 2),
    )

    original = _build(synthetic)
    assert original.exceptAll(with_future).count() == 0
    assert with_future.exceptAll(original).count() == 0
