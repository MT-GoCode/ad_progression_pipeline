import numpy as np
import pandas as pd

from ad_progression_pipeline.components.ingestion.tasks import (
    column_transformers, reshapers)
from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def categorical_ingestion(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = column_transformers.run(train_df, test_df)

    # CONFIRMED WITH PDB at this point that train_df and test_df have similar distributions of CDRSUM
    # note there are 3 rows without proper CDRSUM values in test_df; they aren't transformed properly.

    return reshapers.flatten_add_progression(train_df), reshapers.flatten_add_progression(test_df)


@local_cached_task
def sequential_ingestion(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[np.ndarray]:
    train_df, test_df = column_transformers.run(train_df, test_df)
    return [*reshapers.sequence_ingestion(train_df), *reshapers.sequence_ingestion(test_df)]
