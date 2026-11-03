"""Train a baseline gradient-boosted model and log everything to MLflow.

The model is deliberately plain. The point of the run is the lineage: the
exact Delta version of every input table is attached to the run, so the
training set can be rebuilt bit for bit with ``versionAsOf``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_features.config import Settings
from fraud_features.features import FEATURE_COLUMNS

# HistGradientBoosting models are saved with skops, which refuses to load
# types it has not been told to trust. This is the one type the model uses.
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


@dataclass(frozen=True)
class TrainingConfig:
    test_fraction: float = 0.2
    learning_rate: float = 0.1
    max_iter: int = 200
    max_leaf_nodes: int = 31
    seed: int = 42


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earlier transactions, test on the later ones.

    A random split lets the model learn from transactions that happen after
    the ones it is tested on, which inflates every metric.
    """
    ordered = df.sort_values(["event_ts", "txn_id"], kind="stable")
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Features as float64. Count features are integers in Spark but can be
    NULL (no history yet), and only a float column can carry that as NaN at
    serving time without breaking the logged model signature."""
    return df[list(FEATURE_COLUMNS)].astype("float64")


def evaluate(y_true: np.ndarray, scores: np.ndarray, top_share: float = 0.01) -> dict[str, float]:
    """Ranking metrics. Precision and recall in the top 1% match how a fraud
    team works: a fixed review capacity spent on the highest scores."""
    k = max(1, int(len(scores) * top_share))
    top = np.argsort(-scores, kind="stable")[:k]
    hits = float(y_true[top].sum())
    positives = float(y_true.sum())
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision_at_top_1pct": hits / k,
        "recall_at_top_1pct": hits / positives if positives else 0.0,
        "test_fraud_rate": float(y_true.mean()),
    }


def _experiment_id(settings: Settings) -> str:
    uri = settings.mlflow_tracking_uri
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    existing = mlflow.get_experiment_by_name(settings.mlflow_experiment)
    if existing is not None:
        return existing.experiment_id
    artifacts = Path(settings.mlflow_artifact_root).resolve()
    return mlflow.create_experiment(
        settings.mlflow_experiment, artifact_location=artifacts.as_uri()
    )


def train_and_log(
    training_set: pd.DataFrame,
    lineage: dict[str, str],
    settings: Settings,
    cfg: TrainingConfig | None = None,
) -> str:
    """Fit, evaluate on the later slice, log to MLflow and return the run id."""
    cfg = cfg or TrainingConfig()
    train, test = time_split(training_set, cfg.test_fraction)
    x_train, x_test = feature_matrix(train), feature_matrix(test)
    model = HistGradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        random_state=cfg.seed,
    )
    model.fit(x_train, train["is_fraud"])
    test_scores = model.predict_proba(x_test)[:, 1]
    metrics = evaluate(test["is_fraud"].to_numpy(), test_scores)

    with mlflow.start_run(experiment_id=_experiment_id(settings)) as run:
        mlflow.set_tags(lineage)
        mlflow.log_params(
            {
                **asdict(cfg),
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"features": list(FEATURE_COLUMNS)}, "features.json")
        mlflow.log_input(
            mlflow.data.from_pandas(
                training_set,
                source=lineage.get("training_set_path", "in-memory"),
                name=f"training_set@v{lineage.get('training_set_version', 'na')}",
                targets="is_fraud",
            ),
            context="training",
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(x_test, test_scores),
            input_example=x_test.head(5),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        return run.info.run_id
