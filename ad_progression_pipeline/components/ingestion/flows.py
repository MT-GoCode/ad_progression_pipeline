import numpy as np
import pandas as pd

from ad_progression_pipeline.components.ingestion.tasks import (
    column_transformers,
    reshapers,
)


def categorical_ingestion(df: pd.DataFrame) -> pd.DataFrame:
    df = column_transformers.run(df)
    return reshapers.flatten_add_progression(df)


def sequential_ingestion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = column_transformers.run(df)
    return reshapers.sequence_ingestion(df)
