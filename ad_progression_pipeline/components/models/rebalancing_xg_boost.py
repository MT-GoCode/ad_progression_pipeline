import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from prefect import context, get_run_logger
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ad_progression_pipeline.utils.constants import RANDOM_SEED

from .rebalancing_random_forest import RebalancingRandomForest


class RebalancingXGBoost(RebalancingRandomForest):
    def train(self, df: pd.DataFrame) -> None:
        log = get_run_logger()

        x = df.drop(columns=["progression"])
        y = df["progression"]

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_SEED)),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=RANDOM_SEED,
                        n_estimators=context.hyperparameters["n_estimators"],
                        max_depth=context.hyperparameters["max_depth"],
                        learning_rate=context.hyperparameters["learning_rate"],
                        gamma=context.hyperparameters["gamma"],
                        subsample=0.8,
                        colsample_bytree=0.8,
                        booster="gbtree",
                        objective="binary:logistic",
                    ),
                ),
            ],
        )

        self.model.fit(x, y)
