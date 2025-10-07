Forecast Engine for Business Demand Forecasting Based on Meta Prophet Model.
--------
This is a README snippet for running the Forecast Engine implemented in `forecast_engine.py`.

Prerequisites

1) Create and activate a virtual environment (zsh/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

Quick usage example (interactive):

1. Prepare a pandas DataFrame `df` with monthly data and columns ['ds','y'] where `ds` are monthly period start dates (use freq='MS').

2. Instantiate the engine and run a single model for a series (example):

```python
from forecast_engine import prophet_engine    # import the renamed module (forecast_engine.py)
import pandas as pd

# minimal synthetic series
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=24, freq='MS'),
    'y': [10,12,13,11,15,18,20,17,16,18,19,22,23,21,20,22,25,28,27,26,24,23,22,21]
})

# the engine expects a DataFrame that may also contain grouping dims; for a single series leave ts_dim empty
engine = prophet_engine(ts_data=df, ts_dim=[], param_grid={'changepoint_prior_scale':[0.01, 0.1]})

# run a quick CV for this single series
best_mape, best_result, best_agg = engine.single_model_run(ts_info=pd.Series({'dsn':len(df)}), ts_all=df)
print(best_mape)
```
Capacity
--------

This Forecast Engine is designed for batch processing and can handle large-scale
workloads such as training and evaluating 1,000+ individual forecasting models
in batch, subject to available CPU, memory and time budget. Practical capacity
depends on:

- Available compute (number of CPU cores, RAM per worker)
- Parallelization strategy (single-process, multiprocessing, or distributed)
- Complexity of the model grid (number of hyperparameter combinations)
- Length of each time series and the number of CV windows

On a single high-end VM (multi-core, 32+ GB RAM) you can expect to run
hundreds of models in parallel using multiprocessing; for 1,000+ models a
distributed solution or cloud-based autoscaling is recommended (see roadmap
below).

Improvements & Roadmap
----------------------

Current implementation is a pure-Python, in-memory engine suitable for
prototyping and small-to-medium batch workloads. Recommended improvements to
support production-scale workloads:

1. Refactor to a distributed processing framework (Spark)
     - Move data preparation and per-series model training to Spark DataFrame
         pipelines and use map/foreach patterns for parallelism.
     - Leverage cluster resources to horizontally scale training across many
         executors, making 1,000+ model training runs practical and faster.

2. Containerize and deploy on cloud platforms
     - Package the engine as a Docker image and deploy to managed services
         (AWS Batch / AWS EMR / GCP Dataproc / Azure Synapse) or Kubernetes to
         enable autoscaling and resilient job management.

3. Use a model store and cache
     - Persist fitted models to a model registry or object storage (S3/GCS) to
         avoid re-training and to enable fast batch inference.

4. Improve monitoring and observability
     - Emit metrics (job durations, model MAPE distribution) to a monitoring
         system; log artifacts and test runs to enable reproducible pipelines.

5. Rework forecasting loop for batch inference
     - Separate training and inference code paths so predictions can be served
         quickly from lightweight containers with cached models.

These improvements will allow the engine to scale from exploratory use to
production-grade batch forecasting workloads with predictable performance and
cost controls.
