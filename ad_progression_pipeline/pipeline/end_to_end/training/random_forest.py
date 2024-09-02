import joblib
import optuna
from prefect import context, flow

from ad_progression_pipeline.components.models.tasks import rebalancing_random_forest
from ad_progression_pipeline.pipeline.helpers import context_handler, data, file_handler, optuna_handler


@flow
def random_forest_e2e(config_file : str = "pipeline_config/RandomForest.yaml") -> None:
    context_handler.initialize_context(config_file = config_file)
    output_dir = file_handler.create_output_dir("RandomForest")

    study = optuna.create_study(storage = "sqlite:///" + output_dir + "study.db", direction="maximize")

    @flow
    def objective(trial : optuna.trial.Trial) -> float:
        params = optuna_handler.suggest_hyperparameters(ranges=context.optuna_ranges, trial=trial)
        context_handler.set_hyperparameters(parameters=params)

        train, val, test = data.categorical_data_preparation()

        pipeline = rebalancing_random_forest.train(train)
        val_pred, results = rebalancing_random_forest.infer_and_gather_metrics(pipeline=pipeline, df=val)

        if trial.number == 0 or results["accuracy"] > study.best_value:
            joblib.dump(pipeline, output_dir + "best_model.pkl")
            joblib.dump(results, output_dir + "best_model_metrics.pkl")

        return results["accuracy"]

    study.optimize(objective, n_trials=100)

if __name__ == "__main__":
    random_forest_e2e()
