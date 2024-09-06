from typing import Any

import pandas as pd
from prefect import context

from ad_progression_pipeline.utils.constants import TOTAL_VISITS
from ad_progression_pipeline.utils.prefect import local_cached_task


@local_cached_task
def gen_x(df: pd.DataFrame) -> Any:
    return df.groupby("NACCID").head(context.num_input_visits)


@local_cached_task
def gen_y(df: pd.DataFrame) -> Any:
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
    result.to_csv("flattened_3.csv")
    return result


@local_cached_task
def sequence_ingestion(df: pd.DataFrame) -> pd.DataFrame:
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    df.data.input_data[non_numeric_cols] = df[non_numeric_cols].apply(lambda col: pd.to_numeric(col.astype(str), errors="coerce"))
    x_ = gen_x(df)
    y_ = gen_y(df)
    return x_.drop(columns=["NACCID"]), y_.drop(columns=["NACCID"])
