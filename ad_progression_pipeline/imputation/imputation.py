import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from ..constants import NUMERICAL_FEATURES, ORDINAL_FEATURES, CATEGORICAL_FEATURES
from dataclasses import dataclass

@dataclass
class Imputer:

    input_df: pd.DataFrame
    n_neighbors: int = 50

    def train_knn_imputers(self, columns: list, n_neighbors) -> dict:
        imputer_map = {}

        for feat in columns:
            imputer = KNNImputer(n_neighbors=self.n_neighbors)

            df_feat = self.input_df[[feat]]
            imputer.fit(df_feat)

            imputer_map[feat] = imputer

        return imputer_map

    def train_median_imputers(self, columns: list) -> dict:
        imputer_map = {}

        for col in columns:
            imputer = SimpleImputer(strategy='median')

            df_col = self.input_df[[col]]
            imputer.fit(df_col)

            imputer_map[col] = imputer

        return imputer_map


    def run_imputation(self):

        included_numerical_columns = [_ for _ in NUMERICAL_FEATURES if _ in self.input_df.columns]
        knn_imputer_map = self.train_knn_imputers(included_numerical_columns)

        included_single_val_columns = [_ for _ in ORDINAL_FEATURES + CATEGORICAL_FEATURES if _ in self.input_df.columns]
        median_imputer_map = self.train_median_imputers(included_single_val_columns)

        for col in included_numerical_columns:
            self.input_df[[col]] = knn_imputer_map[col].transform(self.input_df[[col]])

        # Apply Median Imputers to ordinal and categorical columns
        for col in included_single_val_columns:
            self.input_df[[col]] = median_imputer_map[col].transform(self.input_df[[col]])

        return self.input_df
