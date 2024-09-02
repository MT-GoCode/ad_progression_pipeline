import pandas as pd
from prefect import flow

from ad_progression_pipeline.components.imputation.tasks import imputation


@flow
def run_imputation(train_df : pd.DataFrame, test_df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    knn_imputer_map, median_imputer_map = imputation.train(train_df = train_df)
    return imputation.run(train_df = train_df, test_df=test_df, knn_imputer_map = knn_imputer_map, median_imputer_map = median_imputer_map)
