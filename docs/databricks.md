# Optional: Databricks MLflow and Unity Catalog

Everything runs locally by default. The same code can log to a Databricks
workspace and read and write Unity Catalog tables with configuration changes
only.

## MLflow tracking

Authenticate the Databricks CLI (`databricks auth login`), which stores a
profile in `~/.databrickscfg`. Do not put tokens in `.env` or in the repo.
Then point the pipeline at the workspace:

```bash
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT=/Users/<you>/fraud-baseline
make train
```

`train.py` calls `mlflow.create_experiment` only when the experiment does not
exist. On Databricks the artifact location is managed by the workspace, so
`MLFLOW_ARTIFACT_ROOT` is ignored once the experiment exists. Create the
experiment in the UI first if your workspace restricts artifact locations.

To register the model in Unity Catalog, set `mlflow.set_registry_uri("databricks-uc")`
and pass `registered_model_name="<catalog>.<schema>.fraud_baseline"` to
`mlflow.sklearn.log_model`.

## Tables

`Settings.table()` builds paths under `LAKE_ROOT`. Pointing `LAKE_ROOT` at a
Unity Catalog volume (`/Volumes/<catalog>/<schema>/<volume>/lake`) works
unchanged. For managed tables, replace the path-based calls in `storage.py`
with `saveAsTable` / `spark.read.table`, and use `DESCRIBE HISTORY` for the
version numbers. `versionAsOf` time travel works the same way.

## Feature serving

Delta feature tables with `feature_ts` columns map directly onto Databricks
feature tables with a timestamp key, which provide the same point-in-time
lookup used here in `pit.py`. Online serving is out of scope for this repo.
