"""Synthetic card transactions and chargeback labels.

Three fraud patterns are planted, each one detectable by a different feature
family, so the baseline model has something real to learn:

* card testing: a burst of tiny online charges, then a large one (velocity);
* account takeover: large night-time online purchases in risky categories
  (amount and time-since-last features);
* compromised merchants: a few merchants with a raised fraud rate (merchant
  risk, which is only knowable from *past* labels).

Timestamps are timezone-aware UTC so Spark stores the same instant on any
machine, whatever its local zone.

Labels arrive late, as chargebacks do: ``label_ts`` is days or weeks after the
transaction. That delay is what makes label timing a correctness problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from pyspark.sql import DataFrame, SparkSession

TRANSACTION_SCHEMA = (
    "txn_id string, customer_id string, merchant_id string, merchant_category string, "
    "channel string, amount double, event_ts timestamp"
)
LABEL_SCHEMA = "txn_id string, is_fraud int, label_ts timestamp"

# category -> (median amount, lognormal sigma, share of online transactions)
CATEGORIES: dict[str, tuple[float, float, float]] = {
    "grocery": (38.0, 0.5, 0.05),
    "fuel": (52.0, 0.3, 0.0),
    "restaurant": (41.0, 0.5, 0.1),
    "pharmacy": (24.0, 0.6, 0.1),
    "travel": (380.0, 0.7, 0.8),
    "electronics": (240.0, 0.8, 0.6),
    "digital_goods": (14.0, 0.6, 1.0),
}


@dataclass(frozen=True)
class GeneratorConfig:
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    days: int = 120
    customers: int = 2000
    merchants: int = 300
    seed: int = 42
    compromised_merchant_share: float = 0.03
    compromised_fraud_rate: float = 0.08
    card_testing_per_customer_day: float = 0.0015
    takeover_per_customer_day: float = 0.001
    unreported_fraud_share: float = 0.05


@dataclass
class SyntheticData:
    transactions: list[tuple] = field(default_factory=list)
    labels: list[tuple] = field(default_factory=list)

    def to_frames(self, spark: SparkSession) -> tuple[DataFrame, DataFrame]:
        return (
            spark.createDataFrame(self.transactions, TRANSACTION_SCHEMA),
            spark.createDataFrame(self.labels, LABEL_SCHEMA),
        )


@dataclass(frozen=True)
class _Merchant:
    merchant_id: str
    category: str
    compromised: bool


class _Builder:
    """Accumulates raw events; ids are assigned once everything is time-sorted."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.events: list[tuple[datetime, str, _Merchant, str, float, bool]] = []
        names = list(CATEGORIES)
        self.merchants = [
            _Merchant(
                f"M{i:04d}",
                names[i % len(names)],
                bool(self.rng.random() < cfg.compromised_merchant_share),
            )
            for i in range(cfg.merchants)
        ]
        self.by_category = {c: [m for m in self.merchants if m.category == c] for c in names}

    def add(self, ts: datetime, customer: str, merchant: _Merchant, amount: float, fraud: bool):
        online_share = CATEGORIES[merchant.category][2]
        channel = "online" if self.rng.random() < online_share else "card_present"
        self.events.append((ts, customer, merchant, channel, round(amount, 2), fraud))

    def amount(self, category: str, scale: float = 1.0) -> float:
        median, sigma, _ = CATEGORIES[category]
        return float(median * scale * self.rng.lognormal(0.0, sigma))

    def pick(self, category: str) -> _Merchant:
        options = self.by_category[category]
        return options[int(self.rng.integers(len(options)))]


def _legit_day(b: _Builder, customer: str, day: datetime, rate: float, scale: float, favs):
    for _ in range(int(b.rng.poisson(rate))):
        merchant = (
            favs[int(b.rng.integers(len(favs)))]
            if b.rng.random() < 0.8
            else (b.merchants[int(b.rng.integers(len(b.merchants)))])
        )
        # Daytime-heavy: hours drawn around early evening.
        hour = float(np.clip(b.rng.normal(14, 4), 0, 23.99))
        ts = day + timedelta(hours=hour)
        fraud = merchant.compromised and b.rng.random() < b.cfg.compromised_fraud_rate
        b.add(ts, customer, merchant, b.amount(merchant.category, scale), fraud)


def _card_testing(b: _Builder, customer: str, day: datetime) -> None:
    ts = day + timedelta(hours=float(b.rng.uniform(0, 22)))
    for _ in range(int(b.rng.integers(4, 9))):
        ts += timedelta(minutes=float(b.rng.uniform(0.5, 4)))
        b.add(ts, customer, b.pick("digital_goods"), float(b.rng.uniform(0.5, 3.0)), True)
    ts += timedelta(minutes=float(b.rng.uniform(10, 120)))
    b.add(ts, customer, b.pick("electronics"), float(b.rng.uniform(400, 1500)), True)


def _takeover(b: _Builder, customer: str, day: datetime) -> None:
    ts = day + timedelta(hours=float(b.rng.uniform(0, 5)))
    for _ in range(int(b.rng.integers(2, 5))):
        ts += timedelta(minutes=float(b.rng.uniform(5, 60)))
        category = "electronics" if b.rng.random() < 0.6 else "travel"
        b.add(ts, customer, b.pick(category), b.amount(category, 3.0), True)


def generate(cfg: GeneratorConfig) -> SyntheticData:
    b = _Builder(cfg)
    for c in range(cfg.customers):
        customer = f"C{c:05d}"
        rate = float(b.rng.uniform(0.3, 2.5))
        scale = float(b.rng.lognormal(0.0, 0.4))
        favs = [b.merchants[int(i)] for i in b.rng.integers(len(b.merchants), size=8)]
        for d in range(cfg.days):
            day = cfg.start + timedelta(days=d)
            _legit_day(b, customer, day, rate, scale, favs)
            if b.rng.random() < cfg.card_testing_per_customer_day:
                _card_testing(b, customer, day)
            if b.rng.random() < cfg.takeover_per_customer_day:
                _takeover(b, customer, day)
    return _finalise(b)


def _finalise(b: _Builder) -> SyntheticData:
    out = SyntheticData()
    for n, (ts, customer, merchant, channel, amount, fraud) in enumerate(
        sorted(b.events, key=lambda e: (e[0], e[1]))
    ):
        txn_id = f"T{n:08d}"
        out.transactions.append(
            (txn_id, customer, merchant.merchant_id, merchant.category, channel, amount, ts)
        )
        if fraud and b.rng.random() >= b.cfg.unreported_fraud_share:
            # Chargebacks take days to weeks; a long tail runs past a month.
            delay_days = min(1.0 + float(b.rng.gamma(2.0, 7.0)), 90.0)
            out.labels.append((txn_id, 1, ts + timedelta(days=delay_days)))
    return out
