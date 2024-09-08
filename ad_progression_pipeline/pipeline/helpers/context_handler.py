import yaml
from prefect import context, task

from ad_progression_pipeline.components.models import RNN, RebalancingRandomForest, model_interface
from ad_progression_pipeline.pipeline.end_to_end import train

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

    pipeline_map = {
        "train": train.train,
        # "sequence_trainer" :
    }
    model_map = {
        "Random Forest": {"model": RebalancingRandomForest, "model_type": "binary"},
        "RNN": {"model": RNN, "model_type": "sequence"},
    }

    context.pipeline = pipeline_map[config["pipeline"]]
    context.data_dir = config["data_dir"]
    context.num_input_visits = config["num_input_visits"]
    context.optuna_ranges = config["optuna_ranges"]
    context.model_type = model_map[config["model"]]["model_type"]
    context.model = model_interface.instantiate_model(model_map[config["model"]]["model"])


@task
def set_hyperparameters(parameters: dict) -> None:
    context.hyperparameters = parameters
