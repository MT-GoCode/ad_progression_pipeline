import optuna
from prefect import context, flow, get_run_logger
from sklearn.model_selection import train_test_split

from ad_progression_pipeline.components.ingestion.flows import categorical_ingestion, sequential_ingestion
from ad_progression_pipeline.components.models.model_interface import ModelInterface
from ad_progression_pipeline.pipeline.helpers import cli, context_handler, data, files, optuna_
from ad_progression_pipeline.utils.constants import RANDOM_SEED


@flow
def train() -> None:
    log = get_run_logger()
    model: ModelInterface
    model = context.model

    output_dir = files.create_train_output_dir(model.__class__.__name__)

    study = optuna.create_study(storage="sqlite:///" + output_dir + "study.db", direction="maximize")

    @flow
    def objective(trial: optuna.trial.Trial) -> float:
        params = optuna_.suggest_hyperparameters(ranges=context.optuna_ranges, trial=trial)
        context_handler.set_hyperparameters(parameters=params)

        train_data, test_data = data.read_impute_select()
        # will return data w/ NACCID

        if context.model_type == "binary":
            train_data = categorical_ingestion(df=train_data)
            test_data = categorical_ingestion(df=test_data)
            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=RANDOM_SEED)
            model.train(train_data)
            results = model.infer_and_gather_metrics(val_data)

        elif context.model_type == "sequence":
            train_input_matrix, train_output_matrix = sequential_ingestion(df=train_data)
            test_input_matrix, test_input_matrix = sequential_ingestion(df=test_data)

            import pdb

            pdb.set_trace()

            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=RANDOM_SEED)

            import pdb

            pdb.set_trace()
            train_input_matrix, val_input_matrix, train_output_matrix, val_output_matrix = train_test_split(
                train_input_matrix,
                train_output_matrix,
                test_size=0.2,
                random_state=RANDOM_SEED,
            )
            model.train(train_input_matrix, train_output_matrix)
            results = model.infer_and_gather_metrics(train_input_matrix, train_output_matrix)

        objective_value = results["balanced_accuracy"] if context.model_type == "binary" else results["mse"]

        if trial.number == 0 or objective_value > study.best_value:
            model.serialize(output_dir)
        return objective_value

    study.optimize(objective, n_trials=100)
