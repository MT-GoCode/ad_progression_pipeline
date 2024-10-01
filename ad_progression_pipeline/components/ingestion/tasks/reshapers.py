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


def progression_sliding_window(cdrsum_matrix: np.ndarray) -> np.ndarray:
    """Calculates progression for each NACCID based on sliding window logic.

    Args:
        cdrsum_matrix (np.ndarray): A 2D array with dimensions (number of NACCIDs, TOTAL_VISITS)
                                     where each row represents the CDRGLOB values for a single NACCID.

    Returns:
        np.ndarray: A 1D array with dimensions (number of NACCIDs, 1) containing 0 or 1.
    """
    # Initialize a list to store the progression status for each subject
    progression_list = []

    # Assuming cdrsum_matrix is a list of lists, where each inner list contains TOTAL_VISITS CDRSUM values for a subject
    for cdrsum in cdrsum_matrix:  # For each subject's sequence of CDRSUM values
        # List to store progression status for each sliding window of size 3
        progressor_all_temp_list = []

        # Total number of windows is total visits minus window size plus 1
        num_windows = len(cdrsum) - 2  # For TOTAL_VISITS visits, num_windows = 5

        # Iterate over each possible starting index of the sliding window
        for i in range(num_windows):  # i from 0 to 4 inclusive (5 windows)
            # Extract current window of size 3 starting at index i
            current_window = cdrsum[i : i + 3]

            # Extract next values for each position in the current window
            # If there is no next value, use the last value (mimicking R's lead with default)
            next_window = []
            for j in range(i + 1, i + 4):
                if j < len(cdrsum):
                    next_window.append(cdrsum[j])
                else:
                    next_window.append(cdrsum[-1])  # Use last value if index is out of bounds

            # Check for initial increase between current and next window
            initial_increase = [curr < next_ for curr, next_ in zip(current_window, next_window)]

            # Check if all values are stable or increasing
            is_stable_or_increasing = [curr <= next_ for curr, next_ in zip(current_window, next_window)]

            # Determine progression status for this window
            progressor_all_temp = any(initial_increase) and all(is_stable_or_increasing)

            # Store the progression status for this window
            progressor_all_temp_list.append(progressor_all_temp)

        # After checking all windows, determine overall progression status for the subject
        progressor_all = 1 if any(progressor_all_temp_list) else 0

        # Append the result to the progression list
        progression_list.append(progressor_all)

    return np.array(progression_list).reshape(-1, 1)


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
                        f"{col}_{i + 1}": x[col].iloc[i]
                        for col in [feat for feat in x_.columns if feat not in CONSTANT_COLUMNS]
                        for i in range(len(x))
                    },
                },
            ),
        )
        .reset_index()
    )

    grouped = df.groupby("NACCID")["CDRSUM"].apply(lambda x: list(x.values[:TOTAL_VISITS])).reset_index()
    cdrsum_matrix = np.array([x + [x[-1]] * (TOTAL_VISITS - len(x)) if len(x) < TOTAL_VISITS else x[:TOTAL_VISITS] for x in grouped["CDRSUM"]])
    progression_array = progression_sliding_window(cdrsum_matrix)
    naccid_progression_df = pd.DataFrame({"NACCID": df["NACCID"].unique(), "progression": progression_array.flatten()})

    result = pd.merge(result, naccid_progression_df, on="NACCID", how="left").drop(columns="NACCID")

    # Using PDB, I confirmed here that both test and train, when passed into this function, have 154 columns.

    return result.loc[:, ~result.columns.str.contains("NACCID")]


@local_cached_task
def sequence_ingestion(df: pd.DataFrame) -> list[np.ndarray]:
    # NOTE: CDRSUM is the second to last column. Once NACCID is dropped, it will be the last column.
    # This will be helpful info for RNN's infer_and_train function

    x_ = gen_x(df)
    y_ = gen_y(df)

    # INPUT DATA
    input_df = x_.sort_values("NACCID").drop(columns=["NACCID"])

    timesteps = context.num_input_visits
    num_samples = int(len(input_df) / timesteps)
    num_features = len(input_df.columns)

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

    return [input_matrix, output_matrix]
