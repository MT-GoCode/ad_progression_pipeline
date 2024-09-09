import pandas as pd
from prefect import get_run_logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ad_progression_pipeline.utils.constants import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)
from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def run(df: pd.DataFrame) -> pd.DataFrame:
    log = get_run_logger()
    add_later = df[["CDRSUM", "NACCID"]]

    df = df.drop(columns=["CDRSUM", "NACCID"])

    cat_feats = [x for x in CATEGORICAL_FEATURES if x in df.columns]
    ord_feats = [x for x in ORDINAL_FEATURES if x in df.columns]
    num_feats = [x for x in NUMERICAL_FEATURES if x in df.columns]

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
            ("encoder", OrdinalEncoder()),
        ],
    )
    log.info("comparing what columns we have to transform vs total columns")
    log.info([_ for _ in df.columns if _ not in cat_feats + num_feats + ord_feats])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("ord", ordinal_transformer, ord_feats),
            ("cat", categorical_transformer, cat_feats),
        ],
        verbose_feature_names_out=False,
    )

    preprocessor.fit(df)
    transformed_data = preprocessor.transform(df)

    # new columns consists of the new one hot encoded columns... and everything else as before

    new_columns = (
        num_feats
        + ord_feats
        + list(
            preprocessor.transformers_[2][1]
            .named_steps["encoder"]
            .get_feature_names_out(cat_feats),
        )
    )

    df = pd.DataFrame(transformed_data, columns=new_columns)

    df["CDRSUM"] = add_later["CDRSUM"]
    df["NACCID"] = add_later["NACCID"]

    return df
