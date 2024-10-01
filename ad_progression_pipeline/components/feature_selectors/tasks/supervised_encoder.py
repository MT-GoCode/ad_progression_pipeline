import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def train(df: pd.DataFrame, top_x: int, epochs: int = 50) -> Model:
    X = df.drop(["CDRSUM", "NACCID"], axis=1)
    y = df["CDRSUM"]

    # Encode y to integers starting from 0

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # One-hot encode y
    from tensorflow.keras.utils import to_categorical

    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    inputs = Input(shape=(X.shape[1],))

    # Define the model architecture
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    embeddings = Dense(top_x, activation="relu")(x)  # Embedding layer
    outputs = Dense(num_classes, activation="softmax")(embeddings)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X, y_categorical, epochs=epochs, batch_size=16)

    return model


@local_cached_task
def apply(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    X = df.drop(["CDRSUM", "NACCID"], axis=1)

    layer_output_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    embeddings = layer_output_model.predict(X, verbose=0)

    new_df = pd.DataFrame(embeddings)
    new_df["NACCID"] = df.reset_index()["NACCID"]
    new_df["CDRSUM"] = df.reset_index()["CDRSUM"]

    # CONFIRMED with PDB here that embeddings look fine, with NACCID and CDRSUM at the end.

    return new_df
