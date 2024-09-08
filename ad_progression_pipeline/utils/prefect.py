from datetime import timedelta
from functools import wraps
from typing import Any

from prefect import context, flow, task
from prefect.filesystems import LocalFileSystem
from prefect.tasks import task_input_hash

local_fs = LocalFileSystem(basepath="data/prefect_cache/")


def local_cached_task(func: Any) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1), result_storage=local_fs)(func)(*args, **kwargs)

    return wrapper
