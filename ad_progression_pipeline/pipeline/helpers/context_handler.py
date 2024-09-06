import yaml
from prefect import context, task

from ad_progression_pipeline.components.models import RebalancingRandomForest, model_interface

"""
CONTEXT:

static variables
- data_dir : parent data directory for train/test set, optuna runs, etc.
- num_input_viists : self explanatory
- optuna ranges : the allowed ranges for optuna parameters

dynamic variables
- hyperparameters : an object containing optuna's latest "suggests"
- model_artifacts : an object containing imputers, feature selections, trained weights, etc. for reproducible models.

"""


@task
def initialize_context(config_file: str) -> None:
    with open(config_file) as file:
        config = yaml.safe_load(file)

    model_map = {
        "Random Forest": RebalancingRandomForest,
    }

    context.data_dir = config["data_dir"]
    context.num_input_visits = config["num_input_visits"]
    context.optuna_ranges = config["optuna_ranges"]
    context.model = model_interface.instantiate_model(model_map[config["model"]])


@task
def set_hyperparameters(parameters: dict) -> None:
    context.hyperparameters = parameters
