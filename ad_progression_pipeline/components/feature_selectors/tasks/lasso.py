import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def train(df: pd.DataFrame, top_x: int) -> list:
    x = df.drop(columns=["CDRSUM", "NACCID"])
    y = df["CDRSUM"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x)

    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train_scaled, y)

    important_columns = np.argsort(np.abs(lasso.coef_))[-top_x:]
    important_features = x.columns[important_columns]

    return important_features.tolist()


@local_cached_task
def apply(df: pd.DataFrame, features: list) -> pd.DataFrame:
    return df[["NACCID", *features, "CDRSUM"]]
