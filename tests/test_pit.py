from datetime import datetime

from fraud_features.pit import point_in_time_join

EVENTS = "txn_id string, customer_id string, event_ts timestamp"
FEATURES = "customer_id string, feature_ts timestamp, score double"


def _join(spark, events, features):
    out = point_in_time_join(
        spark.createDataFrame(events, EVENTS),
        spark.createDataFrame(features, FEATURES),
        ["customer_id"],
        "event_ts",
        "txn_id",
        "feature_ts_used",
    )
    return {r.txn_id: (r.score, r.feature_ts_used) for r in out.collect()}


def test_picks_latest_snapshot_at_or_before_the_event(spark):
    d = lambda day, hour=0: datetime(2026, 1, day, hour)  # noqa: E731
    events = [("t1", "c1", d(2, 12)), ("t2", "c1", d(3, 0)), ("t3", "c2", d(5))]
    features = [
        ("c1", d(1), 1.0),
        ("c1", d(2), 2.0),
        ("c1", d(3), 3.0),  # exactly at t2's timestamp: allowed
        ("c1", d(4), 4.0),  # after every c1 event: must never be used
        ("c2", d(9), 9.0),  # only in the future for t3
    ]

    result = _join(spark, events, features)

    assert result == {
        "t1": (2.0, d(2)),
        "t2": (3.0, d(3)),
        "t3": (None, None),  # no snapshot yet: kept, with NULL features
    }


def test_future_snapshots_do_not_change_the_result(spark):
    events = [("t1", "c1", datetime(2026, 1, 2))]
    past = [("c1", datetime(2026, 1, 1), 1.0)]
    future = [("c1", datetime(2026, 1, d), 100.0 + d) for d in range(3, 10)]
    assert _join(spark, events, past) == _join(spark, events, past + future)


def test_one_output_row_per_event(spark):
    events = [(f"t{i}", "c1", datetime(2026, 1, 10)) for i in range(5)]
    features = [("c1", datetime(2026, 1, d), float(d)) for d in range(1, 10)]
    assert len(_join(spark, events, features)) == 5
