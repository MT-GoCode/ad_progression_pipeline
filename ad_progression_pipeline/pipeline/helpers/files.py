import os
import pickle
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from prefect import context, task


@task
def create_train_output_dir(name: str) -> str:
    timestamp = datetime.now().strftime("%m-%d %H-%M")
    path = context.data_dir + "/pipeline_results/train/" + name + "_" + timestamp + "/"
    os.makedirs(path)
    return path


def save_to_cache(content: any) -> str:
    path = context.data_dir + "/pipeline_results/object_storage"

    if not filename:
        filename = f"{uuid.uuid4()}.pkl"
    file_path = os.path.join(path, filename)

    with open(file_path, "wb") as f:
        pickle.dump(content, f)


def load_from_cache(path: str) -> any:
    return pickle.load(path)


def cache_dataframes(func: Callable) -> Callable:
    """Decorator to automatically handle DataFrame caching for Prefect tasks."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Check for DataFrame arguments and cache them
        new_args = [save_to_cache(arg, f"arg_{i}") if isinstance(arg, pd.DataFrame) else arg for i, arg in enumerate(args)]
        new_kwargs = {k: save_to_cache(v, f"kwarg_{k}") if isinstance(v, pd.DataFrame) else v for k, v in kwargs.items()}

        # Run the task
        result = func(*new_args, **new_kwargs)

        # Check for DataFrame result and cache it
        if isinstance(result, tuple):
            return tuple(save_to_cache(res, f"res_{i}") if isinstance(res, pd.DataFrame) else res for i, res in enumerate(result))
        if isinstance(result, pd.DataFrame):
            return save_to_cache(result, "result")
        return result

    return wrapper
