import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ad_progression_pipeline.utils.constants import (CATEGORICAL_FEATURES,
                                                     NUMERICAL_FEATURES,
                                                     ORDINAL_FEATURES)
from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def run(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # QUESTION TO RESOLVE: Should I train preprocessor only on train first, or on train and test values?
    # There will be some unaccounted values in test_df if I only train on train_df

    # METHOD 1 -> TRAIN ON TRAIN_DF + TEST_DF
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    add_later = full_df[["CDRSUM", "NACCID"]].copy()

    preprocessor = train(full_df)
    full_df = apply(full_df, preprocessor)

    full_df[["CDRSUM", "NACCID"]] = add_later[["CDRSUM", "NACCID"]]

    train_df_preprocessed = full_df[: len(train_df)].copy()
    test_df_preprocessed = full_df[len(train_df) :].copy()

    return train_df_preprocessed, test_df_preprocessed


@local_cached_task
def train(df: pd.DataFrame) -> ColumnTransformer:
    cat_feats = [x for x in CATEGORICAL_FEATURES if x in df.columns.values]
    ord_feats = [x for x in ORDINAL_FEATURES if x in df.columns.values]
    num_feats = [x for x in NUMERICAL_FEATURES if x in df.columns.values]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ],
    )
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ],
    )
    ordinal_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("ord", ordinal_transformer, ord_feats),
            ("cat", categorical_transformer, cat_feats),
        ],
        verbose_feature_names_out=False,
    )

    preprocessor.fit(df)
    return preprocessor


def apply(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    transformed_data = preprocessor.transform(df)

    cat_feats = [x for x in CATEGORICAL_FEATURES if x in df.columns.values]
    ord_feats = [x for x in ORDINAL_FEATURES if x in df.columns.values]
    num_feats = [x for x in NUMERICAL_FEATURES if x in df.columns.values]

    new_columns = num_feats + ord_feats + list(preprocessor.named_transformers_["cat"].steps[0][1].get_feature_names_out(cat_feats))
    return pd.DataFrame(transformed_data, columns=new_columns)
