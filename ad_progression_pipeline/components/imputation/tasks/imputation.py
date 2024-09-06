import pandas as pd
from prefect import get_run_logger
from sklearn.impute import KNNImputer, SimpleImputer

from ad_progression_pipeline.utils.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ORDINAL_FEATURES
from ad_progression_pipeline.utils.prefect import local_cached_task


def train_knn_imputers(input_df: pd.DataFrame, columns: list, n_neighbors: int) -> dict:
    imputer_map = {}

    for feat in columns:
        imputer = KNNImputer(n_neighbors=n_neighbors)

        df_feat = input_df[[feat]]
        imputer.fit(df_feat)

        imputer_map[feat] = imputer

    return imputer_map


def train_median_imputers(input_df: pd.DataFrame, columns: list) -> dict:
    imputer_map = {}

    for col in columns:
        imputer = SimpleImputer(strategy="median")

        df_col = input_df[[col]]
        imputer.fit(df_col)

        imputer_map[col] = imputer

    return imputer_map


@local_cached_task
def train(train_df: pd.DataFrame) -> tuple[dict, dict]:
    logger = get_run_logger()
    logger.info("Training Imputers...")
    included_numerical_columns = [_ for _ in NUMERICAL_FEATURES if _ in train_df.columns]
    knn_imputer_map = train_knn_imputers(train_df, included_numerical_columns, n_neighbors=50)

    included_single_val_columns = [_ for _ in ORDINAL_FEATURES + CATEGORICAL_FEATURES if _ in train_df.columns]
    median_imputer_map = train_median_imputers(train_df, included_single_val_columns)

    return knn_imputer_map, median_imputer_map


@local_cached_task
def run(train_df: pd.DataFrame, test_df: pd.DataFrame, knn_imputer_map: dict, median_imputer_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in [train_df, test_df]:
        for col in knn_imputer_map:
            df[[col]] = knn_imputer_map[col].transform(df[[col]])

        for col in median_imputer_map:
            df[[col]] = median_imputer_map[col].transform(df[[col]])

    return train_df, test_df
