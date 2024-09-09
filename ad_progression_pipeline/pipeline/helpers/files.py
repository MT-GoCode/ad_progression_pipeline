import os
from datetime import datetime

from prefect import context, task


@task
def create_train_output_dir(name: str) -> str:
    timestamp = datetime.now().strftime("%m-%d %H-%M")
    path = context.data_dir + "/pipeline_results/train/" + name + "_" + timestamp + "/"
    os.makedirs(path)
    return path
