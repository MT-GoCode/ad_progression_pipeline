import optuna
from prefect import context, flow

from ad_progression_pipeline.components.models.model_interface import ModelInterface
from ad_progression_pipeline.pipeline.helpers import cli, context_handler, data, file_handler, optuna_handler


@flow
def train(model: ModelInterface) -> None:
    output_dir = file_handler.create_output_dir(model.__class__.__name__)

    study = optuna.create_study(storage="sqlite:///" + output_dir + "study.db", direction="maximize")

    @flow
    def objective(trial: optuna.trial.Trial) -> float:
        params = optuna_handler.suggest_hyperparameters(ranges=context.optuna_ranges, trial=trial)
        context_handler.set_hyperparameters(parameters=params)

        train_data, val_data, test_data = data.categorical_data_preparation()

        model.train(train_data)

        results = model.infer_and_gather_metrics(val_data)

        if trial.number == 0 or results["accuracy"] > study.best_value:
            model.serialize(output_dir)

        return results["accuracy"]

    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    args = cli.parse_args()
    context_handler.initialize_context(args.config)
    train(context.model)
