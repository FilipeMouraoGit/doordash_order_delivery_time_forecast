import logging
from typing import List
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO)


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, categorical_columns: List, numerical_columns: List):
        """
        Initialize the class with  a dataframe with columns that have rows with nan values, the dictionary
        of methods to be used defines which method is going to be used for each column, ex:
        {'column_1': 'fill with new category', 'column_2': 'fill with median', 'column_3': 'fill with cluster median'}
        """
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.numerical_columns_scaler = {}
        self.raw_data = data.copy()
        self.scaled_data = data.copy()
        self.feature_processed_data = data.copy()
        self.feature_transformer = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(sparse_output=False), categorical_columns),
            ('num', MinMaxScaler(), numerical_columns)
        ])

    def _percentile_scaler_training_data(self, lower_percentile, upper_percentile):
        for column in self.numerical_columns:
            p_min = np.percentile(self.raw_data[column], lower_percentile)
            p_max = np.percentile(self.raw_data[column], upper_percentile)
            self.numerical_columns_scaler[column] = {'min_value': p_min, 'max_value': p_max}
            self.scaled_data[column] = np.clip(self.raw_data[column], p_min, p_max)
        return self.scaled_data

    def feature_transform_training_data(self, lower_percentile=0, upper_percentile=99):
        self._percentile_scaler_training_data(lower_percentile, upper_percentile)
        feature_data = self.feature_transformer.fit_transform(self.scaled_data)
        self.feature_processed_data = pd.DataFrame(
            feature_data,
            columns=self.feature_transformer.get_feature_names_out()
        )
        return self.feature_processed_data

    def feature_transform_test_data(self, test_data):
        test_data_processed = test_data.copy()
        for column in self.numerical_columns:  # Scale the numerical columns
            p_min = self.numerical_columns_scaler[column]['min_value']
            p_max = self.numerical_columns_scaler[column]['max_value']
            test_data_processed[column] = np.clip(test_data_processed[column], p_min, p_max)
        test_transformed_data = self.feature_transformer.transform(test_data_processed)
        return test_transformed_data
