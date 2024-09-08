import pandas as pd
from prefect import context, flow

from ad_progression_pipeline.components.feature_selectors.tasks import lasso, supervised_encoder

selector_map = {"LASSO": lasso, "supervised encoder": supervised_encoder}


def run_feature_selection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector = selector_map[context.hyperparameters["feature_selector"]]
    results = selector.train(df=train_df, top_x=context.hyperparameters["num_features_selected"])
    return selector.apply(train_df, results), selector.apply(test_df, results)
