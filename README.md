# First Time Run

1.
`git clone` this

2. Use correct python version.

install python 3.10 executable, then run these:

`cd ad_progression_pipeline`

`poetry env use <path to python 3.10 executable>`

2. Install dependencies

`poetry install`

Deactivate any environments you have, like venv or conda envs.

`poetry shell` to open a virtual environment with the dependencies. Otherwise, you will need to prefix each command with `poetry run`

If you're using VSCode, it is handy to set the correct interpreter path to `which python` to have syntax highlighting

3. Setup prefect

`prefect cloud login` (May need to create a new account, should be quick)

3. Try running an end-to-end pipeline

`python -m ad_progression_pipeline.pipeline.end_to_end.training.categorical_trainer --config sample_configs/RandomForest.yaml`

This random_forest file is a good entry point to explore the codebase.

# File Structure Overview

```
ad_progression_pipeline/

    components/ -> includes code for individual steps of a pipeline, like imputation or running model

    pipeline/ -> includes code that ties together all steps in an end-to-end workflow

        training/ -> training end-to-end workflows

        testing/ -> testing end-to-end workflows

pipeline_config/ -> YAML files for configuring pipelines, like optuna parameter ranges

data/ -> includes all relevant data

    dataset/ -> has the training and testing sets

    optuna_runs/

        <model_name><date_time> -> includes optuna training history, and serialized versions of the model
```

# Development Guide / Notes

Note that while I'm using the new Prefect tool, syntatically, the Python code looks identical. There are just some decorators: `@flow` denotes functions that encompass higher-level pipelines, while `@task` is for lower-level, nitty-gritty functions, with cachable results.

To develop, also run:

`pip install pre-commit ruff mypy prefect types-PyYAML`

`pre-commit install` and then `pre-commit run --all-files` to manually fix pre-commit errors

run `poetry shell` once to create a persistent virtual env with the right python version, or else you gotta do `poetry run XYZ` each time.

`prefect cloud login`

To take advantage of prefect's caching, set

`prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true`

If you keep running into Prefect HTTP errors, run
`prefect config set PREFECT_API_ENABLE_HTTP2=false`

Note that functions not labeled as tasks and flows can still access context. You won't be able to take advantage of caching. Do this when you hit rate limits on parameter passage

You may run into the issue of not being able to import tensorflow even after doing poetry add. note that you must also install tensorflow-intel, tensorflow-macos it seems.

https://github.com/python-poetry/poetry/issues/8271#issuecomment-1712020965

To avoid overloading their API especially when you need to cache results, setup local cache.


If you want to host your own prefect instance,

`prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"`

then start a server at `prefect server start`, then proceed as normal. To return to cloud, use prefect cloud login
