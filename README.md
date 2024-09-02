# First Time Run

1.
`git clone` this

2. Install dependencies

`poetry install`

For some reason, this package may need to be installed separately:

`pip install tensorflow-io-gcs-filesystem && poetry add tensorflow-io-gcs-filesystem==0.31.0`

`pip install prefect`

`prefect cloud login` (May need to create a new account, should be quick)

3. Try running an end-to-end pipeline

`poetry run python -m ad_progression_pipeline.pipelines.training.random_forest`

This random_forest file is a good entry point to explore the codebase.

# File Structure Overview

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


# Development Guide / Notes

Note that while I'm using the new Prefect tool, syntatically, the Python code looks identical. There are just some decorators: `@flow` denotes functions that encompass higher-level pipelines, while `@task` is for lower-level, nitty-gritty functions, with cachable results.

To develop, also run:

`pip install pre-commit ruff mypy prefect types-PyYAML`

`pre-commit install` and then `pre-commit run --all-files` to manually fix pre-commit errors

run `poetry shell` once to create a persistent virtual env with the right python version, or else you gotta do `poetry run XYZ` each time.

`prefect cloud login`

To take advantage of prefect's caching, set

`prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true`
