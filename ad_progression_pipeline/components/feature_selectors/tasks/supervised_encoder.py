import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from ad_progression_pipeline.components.ingestion.tasks import column_transformers
from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def train(df: pd.DataFrame, epochs: int = 50, top_x: int = 40) -> tuple[any, any]:
    df = column_transformers.run(df)

    X = df.drop("CDRSUM", axis=1)  # Input features
    y = df["CDRSUM"]  # Response variable

    inputs = Input(shape=(X.shape[1],))

    # Add the layers
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(top_x, activation="relu")(x)
    outputs = Dense(y.shape[1], activation="softmax")(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
    )

    model.fit(X, y, epochs=epochs, batch_size=16)

    return model


@local_cached_task
def apply(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    df = column_transformers.run(df)

    X = df.drop("CDRSUM", axis=1)

    last_dense_layer_model = tf.keras.Model(
        inputs=model.input, outputs=model.layers[-2].output,
    )

    output_values = last_dense_layer_model.predict(X, verbose=0)

    new_df = pd.DataFrame(output_values)
    new_df["CDRSUM"] = df.reset_index()["CDRSUM"]

    return new_df
