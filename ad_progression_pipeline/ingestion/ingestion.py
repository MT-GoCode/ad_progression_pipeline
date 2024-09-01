# import pandas as pd
# from ad_progression_pipeline.constants import TOTAL_VISITS, CONSTANT_COLUMNS

# def _X(df):
#     return df.groupby('NACCID').head(num_input_visits)

# def _Y(df):
#     return self.df.groupby('NACCID').tail(TOTAL_VISITS - num_input_visits)

# def progression_map(self):

#     last_cdrsum_X = self.X.groupby('NACCID')['CDRSUM'].last()

#     progression_ = []

#     for naccid, group in self.Y.groupby('NACCID'):
#         if naccid in last_cdrsum_X:
#             last_cdrsum = last_cdrsum_X[naccid]

#             # Check that the sequence of CDRSUM in Y is non-decreasing
#             non_decreasing = all(group['CDRSUM'].iloc[i] >= group['CDRSUM'].iloc[i - 1]
#                     for i in range(1, len(group)))

#             # Check that Y increases in CDRSUM from X
#             has_greater = any(group['CDRSUM'] > last_cdrsum)

#             # Main Progression condition
#             progression = 1 if non_decreasing and has_greater else 0

#             progression_.append((naccid, progression))

#     return pd.DataFrame(progression_, columns=['NACCID', 'progression'])

# def categorical_ingestion(self) -> pd.DataFrame:
#     result = self.X.groupby('NACCID').apply(lambda x: pd.Series({
#         **{col: x[col].iloc[0] for col in CONSTANT_COLUMNS},
#         **{f"{col}_{i+1}": x[col].iloc[i] for col in [feat for feat in self.X.columns if feat not in CONSTANT_COLUMNS] for i in range(len(x))}
#     })).reset_index()

#     result = pd.merge(result, self.progression_map, on='NACCID', how='left').drop(columns='NACCID')
#     result = result.loc[:, ~result.columns.str.contains('NACCID')]
#     result.to_csv("flattened_3.csv")
#     return result


# def sequence_ingestion(self) -> pd.DataFrame:

#     non_numeric_cols = self.data.input_data.select_dtypes(exclude=['number']).columns
#     self.data.input_data[non_numeric_cols] = self.data.input_data[non_numeric_cols].apply(
# lambda col: pd.to_numeric(col.astype(str), errors='coerce'))

#     return self.X.drop(columns=['NACCID']), self.Y.drop(columns=['NACCID'])
