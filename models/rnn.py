from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from typing import Optional, Callable
import functools

@dataclass
class Hyperparameters:
    feature_selector: Optional[Callable] = None
    num_features_selected: Optional[int] = None 
    lstm_units_1: Optional[int] = None
    lstm_units_2: Optional[int] = None
    lstm_dropout_rate_1: Optional[float] = None
    lstm_dropout_rate_2: Optional[float] = None
    dense_dropout_rate: Optional[float] = None
    dense_layer_count: Optional[int] = None
    dense_units: Optional[int] = None
    loss_function: Optional[Callable] = None
    epochs: Optional[int] = None
    optimizer: Optional[Callable] = None
    learning_rate: Optional[float] = None

class RNN_Factory:
    
    def __init__(self, params, visit_inputs):
        self.data = Data()
        self.data.raw = pd.read_csv('data6.csv')
        self.clean_data()

        self.visit_inputs = visit_inputs

        self.params = params
        
    def clean_data(self):
        self.data.clean = self.data.raw.apply(lambda x: x.fillna(x.median()) if x.dtype.kind in 'biufc' else x)

    def data_pipeline(self):
        self.data.input_data = self.data.clean

        self.data.input_data['progression'] = self.data.clean['progression'].astype('category')

        # Feature selection
        selector = functools.partial(self.params.feature_selector, self.params.num_features_selected)
        self.data.input_data = selector(self.data.clean)
        # self.data.input_data = self.data.clean[['CDRSUM', 'SEX'] + top_features]

        # numeric column conversion
        non_numeric_cols = self.data.input_data.select_dtypes(exclude=['number']).columns
        self.data.input_data[non_numeric_cols] = self.data.input_data[non_numeric_cols].apply(lambda col: pd.to_numeric(col.astype(str), errors='coerce'))

        # INPUT DATA
        self.data.input_data = self.data.input_data.drop(columns=['CDRSUM'])
        
        # nice meta-variables
        self.num_samples = len(self.data.input_data) / self.visit_inputs
        self.timesteps = self.visit_inputs
        self.num_features = len(self.data.input_data.columns)
        
        self.data.input_matrix = self.data.input_data.to_numpy()
        self.data.input_matrix = self.data.input_matrix.reshape((int(self.num_samples), int(self.timesteps), int(self.num_features)))

        # create output data
        seven_visits_train_new = pd.read_csv('seven_visits_train_new.csv')
        _ = seven_visits_train_new.copy()
        _['visit_number'] = _.groupby('NACCID').cumcount() + 1
        patients_with_na = _.groupby('NACCID').filter(lambda x: x[['EDUC', 'NACCAGE', 'SEX']].isna().any().any())

        _ = _[~_['NACCID'].isin(patients_with_na['NACCID'].unique())]

        output_data = _.groupby('NACCID').agg(
            cdrsum_4=('CDRSUM', lambda x: x[_['visit_number'] == 4].values[0] if (_['visit_number'] == 4).any() else None),
            cdrsum_5=('CDRSUM', lambda x: x[_['visit_number'] == 5].values[0] if (_['visit_number'] == 5).any() else None),
            cdrsum_6=('CDRSUM', lambda x: x[_['visit_number'] == 6].values[0] if (_['visit_number'] == 6).any() else None),
            cdrsum_7=('CDRSUM', lambda x: x[_['visit_number'] == 7].values[0] if (_['visit_number'] == 7).any() else None)
        ).reset_index()
        self.data.output_data = output_data
        self.data.output_data = self.data.output_data.sort_values('NACCID').fillna(0).drop(columns=['NACCID'])

        # Select the desired columns and convert to a NumPy matrix
        self.data.output_matrix = self.data.output_data[['cdrsum_4', 'cdrsum_5', 'cdrsum_6', 'cdrsum_7']].to_numpy()

    def build_train_model(self, X : pd.DataFrame, Y):



        model = models.Sequential()
        model.add(layers.LSTM(units=self.params.lstm_units_1, return_sequences=True, input_shape=(self.timesteps, self.num_features)))
        model.add(layers.Dropout(rate=self.params.lstm_dropout_rate_1))
        model.add(layers.LSTM(units=self.params.lstm_units_2))
        model.add(layers.Dropout(rate=self.params.lstm_dropout_rate_2))
        for i in range(self.params.dense_layer_count):
            model.add(layers.Dense(units=self.params.dense_units, activation='relu'))
            model.add(layers.Dropout(rate=self.params.dense_dropout_rate))
        model.add(layers.Dense(units=4, activation='linear', name='output'))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            # verbose=1,             # Prints a message when training stops
            restore_best_weights=True
        )

        model.compile(
            loss=self.params.loss_function,  
            optimizer=self.params.optimizer(self.params.learning_rate),
            metrics=['mean_absolute_error'],
        )

        history = model.fit(
            self.data.input_matrix,
            self.data.output_matrix,
            epochs=self.params.epochs, 
            batch_size=256,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stopping]
        )

        return model, history
