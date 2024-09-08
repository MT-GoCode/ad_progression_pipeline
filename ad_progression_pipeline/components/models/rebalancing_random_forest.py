import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from prefect import context, get_run_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler

from ad_progression_pipeline.components.models.tasks.serialize_metrics import serialize_metrics
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

    def infer_and_gather_metrics(self, df: pd.DataFrame) -> dict:
        x = df.drop(columns=["progression"])
        y = df["progression"]

        # Get predictions and probability scores
        y_pred = self.model.predict(x)
        y_proba = self.model.predict_proba(x)[:, 1]

        # Calculate the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc_value = auc(fpr, tpr)

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Generate ROC curve plot and store the ax object
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_value:.2f})")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic")
        ax_roc.legend(loc="lower right")

        # Generate Confusion Matrix plot and store the ax object
        fig_cm, ax_cm = plt.subplots()
        cax = ax_cm.matshow(cm, cmap="Blues")
        fig_cm.colorbar(cax)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")

        smote = self.model.named_steps["smote"]
        X_resampled, y_resampled = smote.fit_resample(x, y)

        self.results = context.hyperparameters
        self.results.update(
            {
                "balanced_accuracy_score": balanced_accuracy_score(y, y_pred),
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average="binary"),
                "recall": recall_score(y, y_pred, average="binary"),
                "f1": f1_score(y, y_pred, average="binary"),
                "original class distribution": f"{Counter(y)!s}",
                "Resampled class distribution": f"{Counter(y_resampled)!s}",
                "roc_auc": ax_roc,  # Store ROC curve ax object here
                "confusion_matrix": ax_cm,  # Store Confusion Matrix ax object here
            },
        )

        return self.results

    def serialize(self, output_dir: str) -> None:
        joblib.dump(self.model, os.path.join(output_dir, "best_model.pkl"))
        serialize_metrics(data=self.results, file_path=os.path.join(output_dir, "best_model_report.md"))

    def deserialize(self, input_dir: str) -> None:
        model_path = os.path.join(input_dir, "best_model.pkl")
        self.model = joblib.load(model_path)
