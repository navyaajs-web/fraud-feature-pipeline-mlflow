"""Runtime settings, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    lake_root: str = "./lake"
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_artifact_root: str = "./mlruns/artifacts"
    mlflow_experiment: str = "fraud-baseline"
    spark_master: str = "local[2]"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            lake_root=os.getenv("LAKE_ROOT", d.lake_root),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", d.mlflow_tracking_uri),
            mlflow_artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", d.mlflow_artifact_root),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", d.mlflow_experiment),
            spark_master=os.getenv("SPARK_MASTER", d.spark_master),
        )

    def table(self, name: str) -> str:
        return str(Path(self.lake_root) / name)
