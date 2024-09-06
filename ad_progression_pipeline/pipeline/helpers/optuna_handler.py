import optuna
from prefect import task


@task
def suggest_hyperparameters(ranges: dict, trial: optuna.trial.Trial) -> dict:
    _ = {}
    if "categorical" in ranges:
        for param, choices in ranges["categorical"].items():
            _[param] = trial.suggest_categorical(param, choices)
    if "int" in ranges:
        for param, range_ in ranges["int"].items():
            _[param] = trial.suggest_int(param, *range_)
    if "float" in ranges:
        for param, range_ in ranges["float"].items():
            _[param] = trial.suggest_float(param, *range_)

    return _
