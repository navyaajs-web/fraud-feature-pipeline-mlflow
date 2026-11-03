PYTHON ?= python
MLFLOW_TRACKING_URI ?= sqlite:///mlruns/mlflow.db

.PHONY: setup features train ui test lint fmt clean

setup:            ## install the package and dev tools into the current Python env
	pip install -e ".[dev]"

features:         ## generate synthetic data and build the Delta feature tables
	$(PYTHON) -m fraud_features.pipeline features

train:            ## build the point-in-time training set, train, log to MLflow (mlruns/)
	$(PYTHON) -m fraud_features.pipeline train

ui:               ## MLflow UI on http://localhost:5000
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --host 127.0.0.1 --port 5000

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

clean:
	rm -rf lake mlruns spark-warehouse metastore_db derby.log .pytest_cache .ruff_cache
