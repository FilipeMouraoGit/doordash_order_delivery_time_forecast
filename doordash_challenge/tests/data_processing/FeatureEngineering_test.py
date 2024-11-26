import unittest
import pandas as pd
import numpy as np
from doordash_challenge.functions.data_processing.FeatureEngineering import FeatureEngineering
from doordash_challenge.functions.data_processing.utils import *


class DataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            'cat_col_1': ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11'],
            'cat_col_2': ['t_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2'],
            'num_col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101],
            'num_col_2': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
        })

        self.feature_selection = FeatureEngineering(
            data=self.raw_data,
            categorical_columns=['cat_col_1', 'cat_col_2'],
            numerical_columns=['num_col_1', 'num_col_2']
        )

    def test_percentile_scaler_training_data_min_max(self):
        expected_result = pd.DataFrame({
            'cat_col_1': ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11'],
            'cat_col_2': ['t_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2'],
            'num_col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101],
            'num_col_2': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
        })
        returned = self.feature_selection._percentile_scaler_training_data(lower_percentile=0, upper_percentile=100)
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_percentile_scaler_training_data_min_p90(self):
        expected_result = pd.DataFrame({
            'cat_col_1': ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11'],
            'cat_col_2': ['t_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2'],
            'num_col_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
            'num_col_2': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 90]
        })
        returned = self.feature_selection._percentile_scaler_training_data(lower_percentile=0, upper_percentile=90)
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_percentile_scaler_training_data_p10_p90(self):
        expected_result = pd.DataFrame({
            'cat_col_1': ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11'],
            'cat_col_2': ['t_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2', 't_3', 't_1', 't_2'],
            'num_col_1': [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
            'num_col_2': [10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 90]
        })
        returned = self.feature_selection._percentile_scaler_training_data(lower_percentile=10, upper_percentile=90)
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_training_data_one_hot_encoder(self):
        expected_result = pd.DataFrame({
            'cat__cat_col_1_c_1':  [1., 0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_2':  [0., 1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_3':  [0., 0., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_4':  [0., 0., 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_5':  [0., 0., 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_6':  [0., 0., 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_7':  [0., 0., 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
            'cat__cat_col_1_c_8':  [0., 0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
            'cat__cat_col_1_c_9':  [0., 0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
            'cat__cat_col_1_c_10': [0., 0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
            'cat__cat_col_2_t_1':  [1., 0., 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            'cat__cat_col_2_t_2':  [0., 1., 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.],
            'cat__cat_col_2_t_3':  [0., 0., 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]
        })
        returned = self.feature_selection.feature_transform_training_data(lower_percentile=0, upper_percentile=100)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_training_data_one_hot_encoder__infrequent_class(self):
        data = pd.DataFrame({
            'cat_col_1': ['t_1', 't_1', 't_1', 't_1', 't_1', 't_1', 't_1', 't_2', 't_2', 't_2', 't_2', 't_2', 't_3'],
            'num_col_1': [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11],
        })
        feature_selection = FeatureEngineering(
                data=data,
                categorical_columns=['cat_col_1'],
                numerical_columns=['num_col_1'],
                min_frequency=0.1
            )
        expected_result = pd.DataFrame({
            'cat__cat_col_1_t_1': [1., 0., 0.],
            'cat__cat_col_1_t_2': [0., 1., 0.],
            'cat__cat_col_1_infrequent_sklearn': [0., 0., 1.],
            'num__num_col_1': [0, 3 / 8, 1]}
        )
        feature_selection.feature_transform_training_data(lower_percentile=10, upper_percentile=90)
        tested_data = pd.DataFrame({'cat_col_1': ['t_1', 't_2', 'c_12'], 'num_col_1': [-5, 5, 30]})
        returned = feature_selection.feature_transform_test_data(tested_data)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)
    def test_feature_transform_training_data_one_hot_encoder__infrequent_multiple_class(self):
        data = pd.DataFrame({
            'cat_col_1': ['t_1', 't_1', 't_1', 't_1', 't_1', 't_1', 't_1', 't_2', 't_2', 't_2', 't_2', 't_2', 't_3'],
            'cat_col_2': ['c_1', 'c_1', 'c_1', 'c_1', 'c_1', 'c_2', 'c_2', 'c_2', 'c_2', 'c_2', 'c_2', 'c_2', 'c_3'],
            'num_col_1': [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11],
        })
        feature_selection = FeatureEngineering(
                data=data,
                categorical_columns=['cat_col_1', 'cat_col_2'],
                numerical_columns=['num_col_1'],
                min_frequency=0.1
            )
        expected_result = pd.DataFrame({
            'cat__cat_col_1_t_1': [1., 0., 0.],
            'cat__cat_col_1_t_2': [0., 1., 0.],
            'cat__cat_col_1_infrequent_sklearn': [0., 0., 1.],
            'cat__cat_col_2_c_1': [1., 0., 0.],
            'cat__cat_col_2_c_2': [0., 1., 0.],
            'cat__cat_col_2_infrequent_sklearn': [0., 0., 1.],
            'num__num_col_1': [0, 3 / 8, 1]}
        )
        feature_selection.feature_transform_training_data(lower_percentile=10, upper_percentile=90)
        tested_data = pd.DataFrame({
            'cat_col_1': ['t_1', 't_2', 'c_12'], 'cat_col_2': ['c_1', 'c_2', 'c_12'], 'num_col_1': [-5, 5, 30]
        })
        returned = feature_selection.feature_transform_test_data(tested_data)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_training_data_min_max(self):
        expected_result = pd.DataFrame({
            'num__num_col_1': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1],
            'num__num_col_2': [0, 0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 1]
        })
        returned = self.feature_selection.feature_transform_training_data(lower_percentile=0, upper_percentile=100)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_training_data_p10_p90(self):
        expected_result = pd.DataFrame({
            'num__num_col_1': [0, 0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1, 1],
            'num__num_col_2': [0, 0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1, 1]
        })
        returned = self.feature_selection.feature_transform_training_data(lower_percentile=10, upper_percentile=90)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_training_data_p20_p80(self):
        expected_result = pd.DataFrame({
            'num__num_col_1': [0, 0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1, 1, 1],
            'num__num_col_2': [0, 0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1, 1, 1]
        })
        returned = self.feature_selection.feature_transform_training_data(lower_percentile=10, upper_percentile=80)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_test_data__expected_values(self):
        tested_data = pd.DataFrame({
            'cat_col_1': ['c_1', 'c_2', 'c_3'],
            'cat_col_2': ['t_1', 't_2', 't_3'],
            'num_col_1': [-5, 5, 30],
            'num_col_2': [30, 60, 300]
        })
        expected_result = pd.DataFrame({
            'cat__cat_col_1_c_1': [1., 0., 0.],
            'cat__cat_col_1_c_2': [0., 1., 0.],
            'cat__cat_col_1_c_3': [0., 0., 1.],
            'cat__cat_col_1_c_4': [0., 0., 0.],
            'cat__cat_col_1_c_5': [0., 0., 0.],
            'cat__cat_col_1_c_6': [0., 0., 0.],
            'cat__cat_col_1_c_7': [0., 0., 0.],
            'cat__cat_col_1_c_8': [0., 0., 0.],
            'cat__cat_col_1_c_9': [0., 0., 0.],
            'cat__cat_col_1_c_10': [0., 0., 0],
            'cat__cat_col_1_c_11': [0., 0., 0],
            'cat__cat_col_2_t_1': [1., 0., 0.],
            'cat__cat_col_2_t_2': [0., 1., 0.],
            'cat__cat_col_2_t_3': [0., 0., 1.],
            'num__num_col_1': [0, 3/8, 1],
            'num__num_col_2': [2/8, 5/8, 1]}
        )
        self.feature_selection.feature_transform_training_data(lower_percentile=10, upper_percentile=90)
        returned = self.feature_selection.feature_transform_test_data(tested_data)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)

    def test_feature_transform_test_data__category_not_seen_in_training(self):
        tested_data = pd.DataFrame({
            'cat_col_1': ['c_1', 'c_2', 'c_12'],
            'cat_col_2': ['t_1', 't_2', 't_3'],
            'num_col_1': [-5, 5, 30],
            'num_col_2': [30, 60, 300]
        })
        expected_result = pd.DataFrame({
            'cat__cat_col_1_c_1': [1., 0., 0.],
            'cat__cat_col_1_c_2': [0., 1., 0.],
            'cat__cat_col_1_c_3': [0., 0., 0.],
            'cat__cat_col_1_c_4': [0., 0., 0.],
            'cat__cat_col_1_c_5': [0., 0., 0.],
            'cat__cat_col_1_c_6': [0., 0., 0.],
            'cat__cat_col_1_c_7': [0., 0., 0.],
            'cat__cat_col_1_c_8': [0., 0., 0.],
            'cat__cat_col_1_c_9': [0., 0., 0.],
            'cat__cat_col_1_c_10': [0., 0., 0],
            'cat__cat_col_1_c_11': [0., 0., 0],
            'cat__cat_col_2_t_1': [1., 0., 0.],
            'cat__cat_col_2_t_2': [0., 1., 0.],
            'cat__cat_col_2_t_3': [0., 0., 1.],
            'num__num_col_1': [0, 3 / 8, 1],
            'num__num_col_2': [2 / 8, 5 / 8, 1]}
        )
        self.feature_selection.feature_transform_training_data(lower_percentile=10, upper_percentile=90)
        returned = self.feature_selection.feature_transform_test_data(tested_data)
        returned = returned[list(expected_result.columns)]
        pd.testing.assert_frame_equal(expected_result, returned)


