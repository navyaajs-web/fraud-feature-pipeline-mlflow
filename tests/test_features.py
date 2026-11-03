from datetime import datetime, timedelta

import pytest

from fraud_features.features import (
    customer_daily_features,
    merchant_daily_features,
    txn_features,
)
from fraud_features.generator import LABEL_SCHEMA, TRANSACTION_SCHEMA

T0 = datetime(2026, 1, 10, 12, 0)


def txn(txn_id, minutes, amount=10.0, customer="c1", merchant="m1", channel="online"):
    return (txn_id, customer, merchant, "grocery", channel, amount, T0 + timedelta(minutes=minutes))


def txns(spark, rows):
    return spark.createDataFrame(rows, TRANSACTION_SCHEMA)


def _by_id(df):
    return {r.txn_id: r for r in df.collect()}


def test_velocity_counts_only_strictly_earlier_transactions(spark):
    rows = [
        txn("a", -60 * 24 * 3, 30.0),  # 3 days earlier
        txn("b", -90, 10.0),  # 90 minutes earlier
        txn("c", -30, 20.0),  # 30 minutes earlier
        txn("d", 0, 60.0),
        txn("e", 0, 5.0, merchant="m2"),  # same second as d: not "before" d
    ]

    d = _by_id(txn_features(txns(spark, rows)))["d"]

    assert (d.cust_txn_count_1h, d.cust_txn_count_24h, d.cust_txn_count_7d) == (1, 2, 3)
    assert d.cust_amount_sum_24h == 30.0
    assert d.amount_to_cust_avg_7d == pytest.approx(60.0 / 20.0)
    assert d.merchant_txn_count_1h == 1
    assert d.feature_ts == d.event_ts


def test_first_transaction_has_no_history(spark):
    a = _by_id(txn_features(txns(spark, [txn("a", 0)])))["a"]
    assert a.secs_since_last_txn is None
    assert (a.cust_txn_count_24h, a.cust_amount_sum_24h, a.amount_to_cust_avg_7d) == (0, 0.0, None)


def test_txn_features_ignore_later_transactions(spark):
    base = [txn("a", -10), txn("b", 0)]
    later = [txn("c", 1), txn("d", 30, merchant="m1")]
    before = _by_id(txn_features(txns(spark, base)))["b"]
    after = _by_id(txn_features(txns(spark, base + later)))["b"]
    assert before == after
    assert after.secs_since_last_txn == 600


def test_customer_snapshot_excludes_its_own_day(spark):
    rows = [txn("a", -60 * 24, 10.0), txn("b", 0, 30.0, merchant="m2"), txn("c", 60 * 24, 50.0)]
    snapshots = customer_daily_features(txns(spark, rows), "2026-01-09", 4).collect()
    by_day = {r.feature_ts.date().day: r for r in snapshots}

    # On 10 Jan only the 9 Jan transaction is visible; 10 Jan's own is not.
    assert by_day[10].cust_txn_count_30d == 1
    assert by_day[11].cust_txn_count_30d == 2
    assert by_day[11].cust_avg_amount_30d == pytest.approx(20.0)
    assert by_day[11].cust_distinct_merchants_30d == 2
    # A snapshot exists for quiet days too, with no division-by-zero.
    assert (by_day[9].cust_txn_count_30d, by_day[9].cust_avg_amount_30d) == (0, None)


def test_merchant_risk_only_counts_labels_known_before_the_snapshot(spark):
    transactions = txns(spark, [txn("f", 0, merchant="m1"), txn("g", 5, merchant="m1")])
    labels = spark.createDataFrame([("f", 1, datetime(2026, 1, 20, 9))], LABEL_SCHEMA)

    rows = merchant_daily_features(transactions, labels, "2026-01-10", 15).collect()
    known = {r.feature_ts.date().day: r.merchant_known_fraud_90d for r in rows}
    risk = {r.feature_ts.date().day: r.merchant_risk for r in rows}

    # The chargeback lands on 20 Jan, so it first counts in the 21 Jan snapshot.
    assert [known[d] for d in (11, 20, 21, 24)] == [0, 0, 1, 1]
    assert risk[21] > risk[20]
