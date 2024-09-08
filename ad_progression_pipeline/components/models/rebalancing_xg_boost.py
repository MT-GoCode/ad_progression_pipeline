# import os
# from collections import Counter

# import joblib
# import matplotlib.pyplot as plt
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from prefect import context, get_run_logger
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, balanced_accuracy_score
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# from ad_progression_pipeline.components.models.tasks.serialize_metrics import serialize_metrics
# from ad_progression_pipeline.utils.constants import RANDOM_SEED

# from .rebalancing_random_forest import RebalancingRandomForest


# class RebalancingXGBoost(RebalancingRandomForest):

#     def train(self, df: pd.DataFrame) -> None:
#         log = get_run_logger()

#         x = df.drop(columns=["progression"])
#         y = df["progression"]

#         self.model = Pipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 ("smote", SMOTE(random_state=RANDOM_SEED)),
#                 (
#                     "classifier",
#                     RandomForestClassifier(
#                         random_state=RANDOM_SEED,
#                         n_estimators=context.hyperparameters["n_estimators"],
#                         max_depth=context.hyperparameters["max_depth"],
#                         criterion=context.hyperparameters["criterion"],
#                     ),
#                 ),
#             ],
#         )

#         self.model.fit(x, y)
