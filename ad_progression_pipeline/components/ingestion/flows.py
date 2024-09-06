import pandas as pd
from prefect import flow

from ad_progression_pipeline.components.ingestion.tasks import reshapers


@flow
def categorical_ingestion(df: pd.DataFrame) -> pd.DataFrame:
    return reshapers.flatten_add_progression(df)


@flow
def sequential_ingestion(df: pd.DataFrame) -> pd.DataFrame:
    return reshapers.sequence_ingestion(df)
