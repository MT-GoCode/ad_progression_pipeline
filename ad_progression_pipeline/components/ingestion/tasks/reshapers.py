from typing import Any

import numpy as np
import pandas as pd
from prefect import context

from ad_progression_pipeline.utils.constants import TOTAL_VISITS
from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def gen_x(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("NACCID").head(context.num_input_visits)


@local_cached_task
def gen_y(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("NACCID").tail(TOTAL_VISITS - context.num_input_visits)


@local_cached_task
def progression_map(df: pd.DataFrame) -> pd.DataFrame:
    last_cdrsum_x = gen_x(df).groupby("NACCID")["CDRSUM"].last()

    progression_ = []

    for naccid, group in gen_y(df).groupby("NACCID"):
        if naccid in last_cdrsum_x:
            last_cdrsum = last_cdrsum_x[naccid]

            # Check that the sequence of CDRSUM in Y is non-decreasing
            non_decreasing = all(group["CDRSUM"].iloc[i] >= group["CDRSUM"].iloc[i - 1] for i in range(1, len(group)))

            # Check that Y increases in CDRSUM from X
            has_greater = any(group["CDRSUM"] > last_cdrsum)

            # Main Progression condition
            progression = 1 if non_decreasing and has_greater else 0

            progression_.append((naccid, progression))

    return pd.DataFrame(progression_, columns=["NACCID", "progression"])


@local_cached_task
def flatten_add_progression(df: pd.DataFrame) -> pd.DataFrame:
    # fix this
    CONSTANT_COLUMNS: Any
    CONSTANT_COLUMNS = []

    x_ = gen_x(df)
    result = (
        x_.groupby("NACCID")
        .apply(
            lambda x: pd.Series(
                {
                    **{col: x[col].iloc[0] for col in CONSTANT_COLUMNS},
                    **{
                        f"{col}_{i+1}": x[col].iloc[i] for col in [feat for feat in x_.columns if feat not in CONSTANT_COLUMNS] for i in range(len(x))
                    },
                },
            ),
        )
        .reset_index()
    )

    result = pd.merge(result, progression_map(df), on="NACCID", how="left").drop(columns="NACCID")
    result = result.loc[:, ~result.columns.str.contains("NACCID")]

    return result


@local_cached_task
def sequence_ingestion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    progression_map(df)

    x_ = gen_x(df)
    y_ = gen_y(df)

    # INPUT DATA
    input_df = x_.sort_values("NACCID").drop(columns=["NACCID"])

    timesteps = context.num_input_visits
    num_samples = int(len(input_df) / timesteps)
    num_features = len(input_df.columns)

    import pdb

    pdb.set_trace()

    input_matrix = input_df.to_numpy()
    input_matrix = input_matrix.reshape((num_samples, timesteps, num_features))

    # OUTPUT DATA
    output_df = y_.sort_values("NACCID")
    output_df = output_df["CDRSUM"]

    timesteps = TOTAL_VISITS - context.num_input_visits
    num_samples = int(len(output_df) / timesteps)
    num_features = 1  # just CDRSUM

    output_matrix = output_df.to_numpy()
    output_matrix = output_matrix.reshape((num_samples, timesteps, num_features))

    return input_matrix, output_matrix
