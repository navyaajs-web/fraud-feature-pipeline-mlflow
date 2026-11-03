from datetime import timedelta

from fraud_features.generator import GeneratorConfig, generate

SMALL = GeneratorConfig(customers=300, merchants=60, days=30, seed=3)


def test_generation_is_deterministic():
    assert generate(SMALL).transactions == generate(SMALL).transactions


def test_labels_arrive_after_their_transaction():
    data = generate(SMALL)
    event_ts = {t[0]: t[-1] for t in data.transactions}
    assert data.labels
    assert all(timedelta(days=1) <= label_ts - event_ts[txn] for txn, _, label_ts in data.labels)


def test_fraud_is_rare_but_present():
    data = generate(SMALL)
    assert 0.002 < len(data.labels) / len(data.transactions) < 0.05


def test_transactions_are_time_ordered_with_unique_ids():
    txns = generate(SMALL).transactions
    assert [t[-1] for t in txns] == sorted(t[-1] for t in txns)
    assert len({t[0] for t in txns}) == len(txns)
