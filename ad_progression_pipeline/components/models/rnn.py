import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from prefect import context
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import backend as K  # noqa: N812
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from ad_progression_pipeline.components.ingestion.tasks.reshapers import progression_sliding_window
from ad_progression_pipeline.components.models.tasks.serialize_metrics import serialize_metrics
from ad_progression_pipeline.utils.constants import TOTAL_VISITS

from .model_interface import ModelInterface


class RNN(ModelInterface):
    model: models.Sequential

    def train(self: "RNN", **kwargs) -> None:  # noqa: ANN003
        input_matrix: np.ndarray = kwargs["input_matrix"]
        output_matrix: np.ndarray = kwargs["output_matrix"]
        params = context.hyperparameters

        self.model = models.Sequential()
        self.model.add(layers.LSTM(units=params["lstm_units_1"], return_sequences=True, input_shape=(input_matrix.shape[1], input_matrix.shape[2])))
        self.model.add(layers.Dropout(rate=params["lstm_dropout_rate_1"]))
        self.model.add(layers.LSTM(units=params["lstm_units_2"]))
        self.model.add(layers.Dropout(rate=params["lstm_dropout_rate_2"]))
        for _ in range(params["dense_layer_count"]):
            self.model.add(layers.Dense(units=params["dense_units"], activation="relu"))
            self.model.add(layers.Dropout(rate=params["dense_dropout_rate"]))
        self.model.add(layers.Dense(units=TOTAL_VISITS - context.num_input_visits, activation="linear", name="output"))

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
        )

        loss_map = {"weighted_mse": weighted_mse, "linear_weighted_mse": linear_weighted_mse}
        optimizer_map = {"Adam": tf.keras.optimizers.Adam, "RMSprop": tf.keras.optimizers.RMSprop, "Nadam": tf.keras.optimizers.Nadam}

        self.model.compile(
            loss=loss_map[params["loss_function"]],
            optimizer=optimizer_map[params["optimizer"]](params["learning_rate"]),
            metrics=["mean_absolute_error"],
        )
        self.history = self.model.fit(
            input_matrix,
            output_matrix,
            epochs=params["epochs"],
            batch_size=256,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stopping],
        )

    def infer_and_gather_metrics(self: "RNN", **kwargs) -> dict:  # noqa: ANN003
        input_matrix: np.ndarray = kwargs["input_matrix"]
        output_matrix: np.ndarray = kwargs["output_matrix"]

        # Step 1: Compute y_pred
        y_pred = self.model.predict(input_matrix)  # Shape: (678, 4)

        # Extract input_matrix_CDRSUM
        input_matrix_CDRSUM = input_matrix[:, :, -1]  # Shape: (678, 3)

        # Step 2: Create concatenated arrays
        # For predicted outputs

        # CONFIRMED WITH PDB that:
        # Shape of input_matrix_CDRSUM: (678, 3)
        # Shape of y_pred: (678, 4)
        # shape of outpu_matrix: (678, 1)

        concatenated_pred = np.concatenate((input_matrix_CDRSUM, y_pred), axis=1)  # Shape: (678, 7)

        # For true outputs, reshape output_matrix to (678, 4)
        output_matrix_reshaped = output_matrix.reshape(output_matrix.shape[0], output_matrix.shape[1])
        concatenated_true = np.concatenate((input_matrix_CDRSUM, output_matrix_reshaped), axis=1)  # Shape: (678, 7)

        # Step 3: Process with progression_sliding_window
        progression_pred = progression_sliding_window(concatenated_pred)  # Shape: (678, 1)
        progression_true = progression_sliding_window(concatenated_true)  # Shape: (678, 1)

        # Ensure progression labels are of integer type
        progression_pred = progression_pred.astype(int).flatten()
        progression_true = progression_true.astype(int).flatten()

        # Step 4: Compute metrics

        progression_accuracy = accuracy_score(progression_true, progression_pred)
        progression_precision = precision_score(progression_true, progression_pred)
        progression_recall = recall_score(progression_true, progression_pred)
        progression_f1 = f1_score(progression_true, progression_pred)
        progression_balanced_accuracy = balanced_accuracy_score(progression_true, progression_pred)

        # Compute confusion matrix
        cm = confusion_matrix(progression_true, progression_pred)

        # Plot confusion matrix
        fig_cm, ax_cm = plt.subplots()
        cax = ax_cm.matshow(cm, cmap="Blues")
        fig_cm.colorbar(cax)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")

        # Set tick marks
        labels = np.unique(np.concatenate((progression_true, progression_pred)))
        ax_cm.set_xticks(range(len(labels)))
        ax_cm.set_yticks(range(len(labels)))
        ax_cm.set_xticklabels(labels)
        ax_cm.set_yticklabels(labels)

        # Annotate each cell with the numerical counts
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="red")

        # Update results dictionary
        self.results = context.hyperparameters
        self.results.update(
            {
                "weighted mse": self.model.evaluate(input_matrix, output_matrix, verbose=0)[0],
                "linear weighted mse": self.model.evaluate(input_matrix, output_matrix, verbose=0)[1],
                "progression_accuracy": progression_accuracy,
                "progression_precision": progression_precision,
                "progression_recall": progression_recall,
                "progression_f1_score": progression_f1,
                "progression_balanced_accuracy": progression_balanced_accuracy,
                "confusion_matrix": ax_cm,  # Include the Axes object
            },
        )

        return self.results

    def serialize(self, output_dir: str) -> None:
        self.model.save(output_dir + "best_model.h5")
        serialize_metrics(data=self.results, file_path=os.path.join(output_dir, "best_model_report.md"))

    def deserialize(self, input_dir: str) -> None:
        pass


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = K.reshape(y_true, shape=(-1, 4))
    y_pred = K.reshape(y_pred, shape=(-1, 4))

    # Calculate the weights
    weights = K.exp(-K.mean(y_true, axis=-1))

    weights_expanded = K.expand_dims(weights, axis=-1)

    mse = K.mean(K.square(y_pred - y_true), axis=-1)

    weighted_mse = mse * weights_expanded

    return K.mean(weighted_mse, axis=-1)


def linear_weighted_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = K.reshape(y_true, shape=(-1, 4))
    y_pred = K.reshape(y_pred, shape=(-1, 4))

    max_y_true = 18.0

    weights = (max_y_true - K.mean(y_true, axis=-1)) / max_y_true
    weights_expanded = K.expand_dims(weights, axis=-1)
    mse = K.square(y_pred - y_true)
    weighted_mse = mse * weights_expanded
    weighted_mse_mean = K.mean(weighted_mse, axis=-1)

    return K.mean(weighted_mse_mean)
