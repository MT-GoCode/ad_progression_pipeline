import os
from collections import Counter
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from prefect import context, get_run_logger, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ad_progression_pipeline.utils.constants import RANDOM_SEED

from .model_interface import ModelInterface


class RebalancingRandomForest(ModelInterface):
    model: Pipeline

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
                    RandomForestClassifier(
                        random_state=RANDOM_SEED,
                        n_estimators=context.hyperparameters["n_estimators"],
                        max_depth=context.hyperparameters["max_depth"],
                        criterion=context.hyperparameters["criterion"],
                    ),
                ),
            ],
        )

        self.model.fit(x, y)

        smote = self.model.named_steps["smote"]
        X_resampled, y_resampled = smote.fit_resample(x, y)
        log.info(f"Original class distribution:{Counter(y)!s}")
        log.info(f"Resampled class distribution: {Counter(y_resampled)!s}")

    def infer_and_gather_metrics(self, df: pd.DataFrame) -> dict:
        x = df.drop(columns=["progression"])
        y = df["progression"]

        y_pred = self.model.predict(x)
        self.results = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="binary"),
            "recall": recall_score(y, y_pred, average="binary"),
            "f1": f1_score(y, y_pred, average="binary"),
        }
        return self.results

    def serialize(self, output_dir: str) -> None:
        joblib.dump(self.model, output_dir + "best_model.pkl")
        joblib.dump(self.results, output_dir + "best_model_metrics.pkl")

    def deserialize(self, input_dir: str) -> None:
        model_path = os.path.join(input_dir, "best_model.pkl")
        self.model = joblib.load(model_path)
