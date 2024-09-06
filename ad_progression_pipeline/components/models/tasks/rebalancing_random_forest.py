from collections import Counter
from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from prefect import context, get_run_logger, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ad_progression_pipeline.utils.constants import RANDOM_SEED


class RebalancingRandomForest:

    @task
    def train(self : "RebalancingRandomForest", df : pd.DataFrame) -> None:

        log = get_run_logger()

        x = df.drop(columns=["progression"])
        y = df["progression"]

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_SEED,
                n_estimators=context.hyperparameters["n_estimators"],
                max_depth=context.hyperparameters["max_depth"],
                criterion=context.hyperparameters["criterion"],
            )),
        ])

        self.pipeline.fit(x, y)

        smote = self.pipeline.named_steps["smote"]
        X_resampled, y_resampled = smote.fit_resample(x, y)
        log.info(f"Original class distribution:{str(Counter(y))}")
        log.info(f"Resampled class distribution: {str(Counter(y_resampled))}")

    @task
    def infer_and_gather_metrics(self : "RebalancingRandomForest", df : pd.DataFrame) -> tuple[Any, dict]:
        x = df.drop(columns=["progression"])
        y = df["progression"]

        y_pred = self.pipeline.predict(x)

        return y_pred, {
            "accuracy" : accuracy_score(y, y_pred),
            "precision" : precision_score(y, y_pred, average="binary"),
            "recall" : recall_score(y, y_pred, average="binary"),
            "f1" : f1_score(y, y_pred, average="binary"),
        }

    # @task
    # def serialize(self, output_dir) -> Pipeline:
