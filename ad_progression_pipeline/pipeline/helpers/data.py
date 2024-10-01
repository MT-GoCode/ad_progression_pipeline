import pandas as pd
from prefect import context, flow, get_run_logger

from ad_progression_pipeline.components.feature_selectors.flows import run_feature_selection
from ad_progression_pipeline.components.imputation.flows import run_imputation
from ad_progression_pipeline.utils.constants import UNACCOUNTED_COLUMNS


@flow
def read_impute_select() -> tuple[pd.DataFrame, pd.DataFrame]:
    log = get_run_logger()

    # STEP 0
    train = pd.read_csv(context.data_dir + "dataset/training_set.csv", index_col=0)
    test = pd.read_csv(context.data_dir + "dataset/test_set.csv", index_col=0)
    train = train.drop(UNACCOUNTED_COLUMNS, axis=1)
    test = test.drop(UNACCOUNTED_COLUMNS, axis=1)

    # STEP 1: IMPUTATION
    train, test = run_imputation(train_df=train, test_df=test)
    # will need this later. column transformation cannot operate on this as it is not a feature column. But flattening will need this.
    log.info("STEP 1: IMPUTATION COMPLETE")

    # STEP 3: FEATURE SELECTION
    # Note: the only non-numerical column at this point is NACCID. feature selection will first remove it.
    # but in the future, if you get a non-numerical error, you'll have to ensure stuff is numerical first
    # note run_feature_selection and its subfunctions should be labeled as flows, bu
    train, test = run_feature_selection(train, test)
    log.info("STEP 3: FEATURE SELECTION COMPLETE")

    return train, test
