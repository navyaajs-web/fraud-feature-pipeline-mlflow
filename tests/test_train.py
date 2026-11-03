from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

from fraud_features.config import Settings
from fraud_features.features import (
    FEATURE_COLUMNS,
    customer_daily_features,
    merchant_daily_features,
    txn_features,
)
from fraud_features.generator import GeneratorConfig, generate
from fraud_features.train import (
    TrainingConfig,
    evaluate,
    feature_matrix,
    time_split,
    train_and_log,
)
from fraud_features.training_set import build_training_set


def test_time_split_tests_on_the_future():
    df = pd.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(10)],
            "event_ts": pd.date_range("2026-01-01", periods=10),
        }
    ).sample(frac=1, random_state=1)
    train, test = time_split(df, 0.3)
    assert train["event_ts"].max() < test["event_ts"].min()
    assert (len(train), len(test)) == (7, 3)


def test_top_share_metrics():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0] * 10)
    scores = y + np.linspace(0, 0.5, len(y))  # frauds always score higher
    metrics = evaluate(y, scores, top_share=0.1)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision_at_top_1pct"] == 1.0
    assert metrics["recall_at_top_1pct"] == 0.5


def test_end_to_end_run_is_logged_with_lineage(spark, tmp_path):
    cfg = GeneratorConfig(customers=400, merchants=60, days=45, seed=9)
    transactions, labels = generate(cfg).to_frames(spark)
    start = cfg.start.date().isoformat()
    training_set = build_training_set(
        transactions,
        labels,
        txn_features(transactions),
        customer_daily_features(transactions, start, cfg.days),
        merchant_daily_features(transactions, labels, start, cfg.days),
        as_of=datetime(2026, 2, 14),
        maturity_days=10,
    ).toPandas()
    settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlruns/mlflow.db",
        mlflow_artifact_root=str(tmp_path / "mlruns" / "artifacts"),
        mlflow_experiment="test",
    )
    lineage = {"training_set_path": "memory", "training_set_version": "0", "as_of": "2026-02-14"}

    run_id = train_and_log(training_set, lineage, settings, TrainingConfig(max_iter=30))

    run = mlflow.get_run(run_id)
    assert run.data.tags["training_set_version"] == "0"
    assert run.data.params["n_features"] == str(len(FEATURE_COLUMNS))
    # Planted patterns are easy; anything near 0.5 means the features are broken.
    assert run.data.metrics["roc_auc"] > 0.8
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert model.predict_proba(feature_matrix(training_set).head(3)).shape == (3, 2)
