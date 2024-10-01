import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from prefect import context
from xgboost import XGBClassifier

from ad_progression_pipeline.utils.constants import RANDOM_SEED

from .rebalancing_random_forest import RebalancingRandomForest


class RebalancingXGBoost(RebalancingRandomForest):
    def train(self: "RebalancingXGBoost", **kwargs) -> None:  # noqa: ANN003
        df: pd.DataFrame = kwargs["df"]

        x = df.drop(columns=["progression"])
        y = df["progression"]

        self.model = Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_SEED) if context.hyperparameters["smote"] else "passthrough"),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=RANDOM_SEED,
                        n_estimators=context.hyperparameters["n_estimators"],
                        max_depth=context.hyperparameters["max_depth"],
                        learning_rate=context.hyperparameters["learning_rate"],
                        subsample=context.hyperparameters["subsample"],
                        gamma=context.hyperparameters["gamma"],
                        booster=context.hyperparameters["booster"],
                        use_label_encoder=False,  # Needed to avoid a warning
                        eval_metric="logloss",  # Default evaluation metric for binary classification
                    ),
                ),
            ],
        )

        self.model.fit(x, y)
