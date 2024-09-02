import pandas as pd
from prefect import context, flow, get_run_logger
from sklearn.model_selection import train_test_split

from ad_progression_pipeline.components.feature_selectors.flows import run_feature_selection
from ad_progression_pipeline.components.imputation.flows import run_imputation
from ad_progression_pipeline.components.ingestion.flows import categorical_ingestion
from ad_progression_pipeline.utils.constants import RANDOM_SEED, UNACCOUNTED_COLUMNS


@flow
def categorical_data_preparation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    log = get_run_logger()

    # STEP 0
    train = pd.read_csv(context.data_dir + "dataset/training_set.csv", index_col=0)
    test = pd.read_csv(context.data_dir + "dataset/test_set.csv", index_col=0)
    # GLUE CODE : not sure how to impute these columns
    train = train.drop(columns=UNACCOUNTED_COLUMNS)
    test = test.drop(columns=UNACCOUNTED_COLUMNS)

    # STEP 1: IMPUTATION
    train, test = run_imputation(train_df = train, test_df = test)
    train_naccid_col = train["NACCID"]
    test_naccid_col = test["NACCID"]
    # will need this later. column transformation cannot operate on this as it is not a feature column. But flattening will need this.
    log.info("STEP 1: IMPUTATION COMPLETE")

    # STEP 2: COLUMN TRANSFORMATION
    # TODO
    # GLUE CODE : haven't figured out column transformation
    # (required to do feature selection on numerical columns). can only feature select on float columns
    train = train.select_dtypes(include=["number"])
    test = test.select_dtypes(include=["number"])
    log.info("STEP 2: COLUMN TRANSFORMATION COMPLETE")

    # STEP 3: FEATURE SELECTION
    train, test = run_feature_selection(train, test)
    log.info("STEP 3: FEATURE SELECTION COMPLETE")

    # STEP 4: RESHAPING + ADD PROGRESSION COLUMN
    train["NACCID"] = train_naccid_col
    test["NACCID"] = test_naccid_col
    train = categorical_ingestion(df = train)
    test = categorical_ingestion(df = test)
    # categorical ingestion

    # STEP 5: SPLIT TRAIN INTO TRAIN AND VAL
    train, val = train_test_split(train, test_size = 0.2, random_state = RANDOM_SEED)

    return train, val, test
