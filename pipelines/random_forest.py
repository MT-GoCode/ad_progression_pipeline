from ingestion import Ingestor
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score
import io
from PIL import Image
from dataclasses import asdict
import matplotlib.pyplot as plt
import mlflow
from dataclasses import dataclass
from functools import cached_property
from constants import RANDOM_SEED
from parameters.parameters import RandomForestPipelineParameters

@dataclass
class RandomForestPipeline():

    @cached_property
    def input_df():
        ingestor = Ingestor(csv_path="seven_visits_train_imputed_with_data6_temp.csv", num_input_visits=3)
        # normally, would run df = ingestor.categorical_ingestion()
        return pd.read_csv('flattened_3.csv').dropna(axis=1, how='any')

    def objective(self, trial):

        params = RandomForestPipelineParameters()
        params.rebalancer = trial.suggest_categorical('rebalancer', SMOTE, RandomOverSampler)
        params.n_estimators = trial.suggest_int('n_estimators', 50, 500)
        params.max_depth = trial.suggest_int('max_depth', 10, 100)
        params.criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        with mlflow.start_run(run_name=f"Trial #{trial.number}"):
            
            X = self.input_df.drop(columns=['progression'])
            y = self.input_df['progression']

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_SEED)),
                ('classifier', RandomForestClassifier(
                    random_state=RANDOM_SEED,
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                    criterion=params.criterion,
                ))
            ])

            pipeline.fit(X, y)

            smote = pipeline.named_steps['smote']
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print("Original class distribution:", Counter(y))
            print("Resampled class distribution:", Counter(y_resampled))
            y_pred = pipeline.predict(X)
            

