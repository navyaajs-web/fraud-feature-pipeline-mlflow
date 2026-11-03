"""Feature jobs. Every feature row carries the time it became knowable.

The rule throughout: a feature with timestamp ``t`` may only use events (and
labels) that were known strictly before ``t``. Each function below states how
it honours that rule, and the tests check it by adding future data and
asserting nothing changes.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

HOUR = 3600
DAY = 86400

TXN_FEATURES = (
    "cust_txn_count_1h",
    "cust_txn_count_24h",
    "cust_txn_count_7d",
    "cust_amount_sum_24h",
    "amount_to_cust_avg_7d",
    "secs_since_last_txn",
    "merchant_txn_count_1h",
    "is_online",
    "hour_of_day",
    "amount",
)
CUSTOMER_DAILY_FEATURES = (
    "cust_txn_count_30d",
    "cust_avg_amount_30d",
    "cust_distinct_merchants_30d",
)
MERCHANT_DAILY_FEATURES = ("merchant_txn_count_90d", "merchant_known_fraud_90d", "merchant_risk")
FEATURE_COLUMNS = TXN_FEATURES + CUSTOMER_DAILY_FEATURES + MERCHANT_DAILY_FEATURES


def _prior(partition: str, seconds: int) -> Window:
    """Window over the same key's events in the ``seconds`` *before* this one.

    The upper bound is -1 second, not 0: the current transaction, and anything
    at the same second, must not count towards its own velocity.
    """
    return (
        Window.partitionBy(partition)
        .orderBy(F.col("event_ts").cast("long"))
        .rangeBetween(-seconds, -1)
    )


def txn_features(transactions: DataFrame) -> DataFrame:
    """Per-transaction features, as of the moment of authorisation (``feature_ts = event_ts``)."""
    by_customer_time = Window.partitionBy("customer_id").orderBy("event_ts", "txn_id")
    prior_7d_avg = F.avg("amount").over(_prior("customer_id", 7 * DAY))
    return transactions.select(
        "txn_id",
        "customer_id",
        "merchant_id",
        "event_ts",
        F.col("event_ts").alias("feature_ts"),
        F.count("*").over(_prior("customer_id", HOUR)).alias("cust_txn_count_1h"),
        F.count("*").over(_prior("customer_id", DAY)).alias("cust_txn_count_24h"),
        F.count("*").over(_prior("customer_id", 7 * DAY)).alias("cust_txn_count_7d"),
        F.coalesce(F.sum("amount").over(_prior("customer_id", DAY)), F.lit(0.0)).alias(
            "cust_amount_sum_24h"
        ),
        (F.col("amount") / prior_7d_avg).alias("amount_to_cust_avg_7d"),
        (
            F.col("event_ts").cast("long") - F.lag("event_ts").over(by_customer_time).cast("long")
        ).alias("secs_since_last_txn"),
        F.count("*").over(_prior("merchant_id", HOUR)).alias("merchant_txn_count_1h"),
        (F.col("channel") == "online").cast("int").alias("is_online"),
        F.hour("event_ts").alias("hour_of_day"),
        "amount",
    )


def _calendar(keys: DataFrame, key: str, start: str, days: int) -> DataFrame:
    """One row per key per day, so snapshots exist even on days with no activity."""
    dates = F.sequence(F.to_date(F.lit(start)), F.date_add(F.to_date(F.lit(start)), days - 1))
    return keys.select(key).distinct().select(key, F.explode(dates).alias("day"))


def _day_index(col: Column) -> Column:
    return F.datediff(col, F.lit("1970-01-01"))


def customer_daily_features(transactions: DataFrame, start: str, days: int) -> DataFrame:
    """Daily customer snapshot for each day ``d`` from transactions in ``[d-30, d)``.

    ``feature_ts`` is midnight at the start of ``d``: the snapshot only covers
    whole days that have already finished, so it is safe to use for any
    transaction on day ``d``.
    """
    daily = transactions.groupBy("customer_id", F.to_date("event_ts").alias("day")).agg(
        F.count("*").alias("n"),
        F.sum("amount").alias("amount"),
        F.collect_set("merchant_id").alias("merchants"),
    )
    window = (
        Window.partitionBy("customer_id").orderBy(_day_index(F.col("day"))).rangeBetween(-30, -1)
    )
    n_30d = F.coalesce(F.sum("n").over(window), F.lit(0))
    return (
        _calendar(transactions, "customer_id", start, days)
        .join(daily, ["customer_id", "day"], "left")
        .select(
            "customer_id",
            F.col("day").cast("timestamp").alias("feature_ts"),
            n_30d.alias("cust_txn_count_30d"),
            # Spark 4 runs in ANSI mode, where x / 0 raises instead of returning NULL.
            F.when(n_30d > 0, F.sum("amount").over(window) / n_30d).alias("cust_avg_amount_30d"),
            F.size(F.array_distinct(F.flatten(F.collect_list("merchants").over(window)))).alias(
                "cust_distinct_merchants_30d"
            ),
        )
    )


def merchant_daily_features(
    transactions: DataFrame,
    labels: DataFrame,
    start: str,
    days: int,
    prior_rate: float = 0.01,
    prior_weight: float = 200.0,
) -> DataFrame:
    """Daily merchant risk for each day ``d``, using only labels known before ``d``.

    This is the feature most likely to leak: a fraud on 3 March that is only
    charged back on 2 April must not raise the merchant's risk before 2 April.
    So a fraud counts towards day ``d`` only if the transaction is inside the
    90-day window *and* its ``label_ts`` is before ``d``.

    ``merchant_risk`` shrinks the observed rate towards a fixed prior, so a
    merchant with three transactions and one fraud is not scored at 33%. The
    prior is a constant rather than the global rate, which would itself have
    to be computed point in time.
    """
    cal = _calendar(transactions, "merchant_id", start, days).withColumn(
        "feature_ts", F.col("day").cast("timestamp")
    )
    daily = transactions.groupBy("merchant_id", F.to_date("event_ts").alias("day")).count()
    window = (
        Window.partitionBy("merchant_id").orderBy(_day_index(F.col("day"))).rangeBetween(-90, -1)
    )
    txn_counts = cal.join(daily, ["merchant_id", "day"], "left").select(
        "merchant_id",
        "feature_ts",
        F.coalesce(F.sum("count").over(window), F.lit(0)).alias("merchant_txn_count_90d"),
    )

    frauds = transactions.join(labels.where("is_fraud = 1"), "txn_id").select(
        F.col("merchant_id").alias("f_merchant_id"), "event_ts", "label_ts"
    )
    known = (
        cal.join(
            frauds,
            (F.col("merchant_id") == F.col("f_merchant_id"))
            & (F.col("event_ts") >= F.col("feature_ts") - F.expr("INTERVAL 90 DAYS"))
            & (F.col("event_ts") < F.col("feature_ts"))
            & (F.col("label_ts") < F.col("feature_ts")),
        )
        .groupBy("merchant_id", "feature_ts")
        .agg(F.count("*").alias("merchant_known_fraud_90d"))
    )
    fraud_count = F.coalesce(F.col("merchant_known_fraud_90d"), F.lit(0))
    return txn_counts.join(known, ["merchant_id", "feature_ts"], "left").select(
        "merchant_id",
        "feature_ts",
        "merchant_txn_count_90d",
        fraud_count.alias("merchant_known_fraud_90d"),
        (
            (fraud_count + F.lit(prior_rate * prior_weight))
            / (F.col("merchant_txn_count_90d") + F.lit(prior_weight))
        ).alias("merchant_risk"),
    )
