import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.pipeline import Pipeline
from prefect import context
from tensorflow.keras import backend as K  # noqa: N812
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from ad_progression_pipeline.utils.constants import TOTAL_VISITS

from .model_interface import ModelInterface


class RNN(ModelInterface):
    model: Pipeline

    def train(self, input_matrix: np.ndarray, output_matrix: np.ndarray) -> None:
        params = context.hyperparameters

        self.model = models.Sequential()
        self.model.add(
            layers.LSTM(
                units=params["lstm_units_1"],
                return_sequences=True,
                input_shape=(input_matrix.shape[1], input_matrix.shape[2]),
            ),
        )
        self.model.add(layers.Dropout(rate=params["lstm_dropout_rate_1"]))
        self.model.add(layers.LSTM(units=params["lstm_units_2"]))
        self.model.add(layers.Dropout(rate=params["lstm_dropout_rate_2"]))
        for _ in range(params["dense_layer_count"]):
            self.model.add(layers.Dense(units=params["dense_units"], activation="relu"))
            self.model.add(layers.Dropout(rate=params["dense_dropout_rate"]))
        self.model.add(
            layers.Dense(
                units=TOTAL_VISITS - context.num_input_visits,
                activation="linear",
                name="output",
            ),
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
        )

        loss_map = {
            "weighted_mse": self.weighted_mse,
            "linear_weighted_mse": self.linear_weighted_mse,
        }
        optimizer_map = {
            "Adam": tf.keras.optimizers.Adam,
            "RMSprop": tf.keras.optimizers.RMSprop,
            "Nadam": tf.keras.optimizers.Nadam,
        }

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

        return self.model, self.history

    def infer_and_gather_metrics(self, df: pd.DataFrame) -> dict:
        pass

    def serialize(self, output_dir: str) -> None:
        self.model.save(output_dir + "best_model.h5")

    def deserialize(self, input_dir: str) -> None:
        pass

    def weighted_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = K.reshape(y_true, shape=(-1, 4))
        y_pred = K.reshape(y_pred, shape=(-1, 4))

        # Calculate the weights
        weights = K.exp(-K.mean(y_true, axis=-1))

        weights_expanded = K.expand_dims(weights, axis=-1)

        mse = K.mean(K.square(y_pred - y_true), axis=-1)

        weighted_mse = mse * weights_expanded

        return K.mean(weighted_mse, axis=-1)

    def linear_weighted_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = K.reshape(y_true, shape=(-1, 4))
        y_pred = K.reshape(y_pred, shape=(-1, 4))

        max_y_true = 18.0

        weights = (max_y_true - K.mean(y_true, axis=-1)) / max_y_true
        weights_expanded = K.expand_dims(weights, axis=-1)
        mse = K.square(y_pred - y_true)
        weighted_mse = mse * weights_expanded
        weighted_mse_mean = K.mean(weighted_mse, axis=-1)

        return K.mean(weighted_mse_mean)
