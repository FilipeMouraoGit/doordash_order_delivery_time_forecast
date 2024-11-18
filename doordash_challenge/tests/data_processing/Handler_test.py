import unittest
import pandas as pd
import numpy as np
from doordash_challenge.functions.data_processing.Handler import DataHandler
from doordash_challenge.functions.data_processing.Handler import FILL_NA_WITH_NEW_CATEGORY_METHOD, \
    FILL_NA_WITH_MEDIAN_METHOD, FILL_NA_WITH_CLUSTER_MEDIAN_METHOD


class DataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'numerical_column_normal_median_1': [1, 2, 3, 4, 5, np.nan],
            'numerical_column_normal_median_2': [6, 7, 8, 9, 10, np.nan],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2, 4, np.nan, 10, 12, np.nan],
        })
        self.fill_column_methods = {
            'categorical_column_1': FILL_NA_WITH_NEW_CATEGORY_METHOD,
            'categorical_column_2': FILL_NA_WITH_NEW_CATEGORY_METHOD,
            'numerical_column_normal_median_1': FILL_NA_WITH_MEDIAN_METHOD,
            'numerical_column_normal_median_2': FILL_NA_WITH_MEDIAN_METHOD,
            'numerical_column_cluster_median': FILL_NA_WITH_CLUSTER_MEDIAN_METHOD
        }
        self.data_handler = DataHandler(data=self.raw_data, fill_column_methods=self.fill_column_methods)

    def test_init_error(self):
        try:
            fill_column_methods = {'categorical_column_1': 'non_defined_method'}
            data_handler = DataHandler(data=self.raw_data, fill_column_methods=fill_column_methods)
        except Exception as error:
            self.assertEqual(error.args[0], 'Method `non_defined_method` not supported by the Handler')

    def test_na_with_new_category__fill_all_columns(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not informed'],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not informed'],
            'numerical_column_normal_median_1': [1, 2, 3, 4, 5, np.nan],
            'numerical_column_normal_median_2': [6, 7, 8, 9, 10, np.nan],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2, 4, np.nan, 10, 12, np.nan],
        })
        df_returned = self.data_handler.fill_na_with_new_category()
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_na_with_new_category__fill_specific_column(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not available'],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not specified'],
            'numerical_column_normal_median_1': [1, 2, 3, 4, 5, np.nan],
            'numerical_column_normal_median_2': [6, 7, 8, 9, 10, np.nan],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2, 4, np.nan, 10, 12, np.nan],
        })
        _ = self.data_handler.fill_na_with_new_category(
            extra_category_name='not available',
            column='categorical_column_1'
        )
        df_returned = self.data_handler.fill_na_with_new_category(
            extra_category_name='not specified',
            column='categorical_column_2'
        )
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_na_with_new_category__general_fill_then_specific(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not available'],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not specified'],
            'numerical_column_normal_median_1': [1, 2, 3, 4, 5, np.nan],
            'numerical_column_normal_median_2': [6, 7, 8, 9, 10, np.nan],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2, 4, np.nan, 10, 12, np.nan],
        })
        _ = self.data_handler.fill_na_with_new_category(
            extra_category_name='not available'
        )
        df_returned = self.data_handler.fill_na_with_new_category(
            extra_category_name='not specified',
            column='categorical_column_2'
        )
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_fill_na_with_median__fill_all_columns(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'numerical_column_normal_median_1': [1., 2., 3., 4., 5., 3.],
            'numerical_column_normal_median_2': [6., 7., 8., 9., 10., 8.],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2, 4, np.nan, 10, 12, np.nan],
        })
        df_returned = self.data_handler.fill_na_with_median()
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_fill_na_with_median__fill_specific_columns(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'numerical_column_normal_median_1': [1., 2., 3., 4., 5., np.nan],
            'numerical_column_normal_median_2': [6., 7., 8., 9., 10., 8.],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2, 4, np.nan, 10, 12, np.nan],
        })
        df_returned = self.data_handler.fill_na_with_median(column='numerical_column_normal_median_2')
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_fill_na_with_cluster_median(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', np.nan],
            'numerical_column_normal_median_1': [1., 2., 3., 4., 5., np.nan],
            'numerical_column_normal_median_2': [6., 7., 8., 9., 10., np.nan],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2.0, 4.0, 3.0, 10.0, 12.0, 11.0],
        })
        df_returned = \
            self.data_handler.fill_na_with_cluster_median(cluster_columns=['categorical_column_cluster_median'])
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_fill_missing_values_training_data(self):
        df_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not informed'],
            'categorical_column_2': ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'not informed'],
            'numerical_column_normal_median_1': [1., 2., 3., 4., 5., 3.0],
            'numerical_column_normal_median_2': [6., 7., 8., 9., 10., 8.0],
            'categorical_column_cluster_median': ['1', '1', '1', '2', '2', '2'],
            'numerical_column_cluster_median': [2.0, 4.0, 3.0, 10.0, 12.0, 11.0],
        })
        cluster_columns = ['categorical_column_cluster_median']
        df_returned = self.data_handler.fill_missing_values_training_data(cluster_columns)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_fill_missing_values_testing_data__raise_error(self):
        df_test = pd.DataFrame({
            'categorical_column_1': ['type_1', np.nan, 'type_3', 'type_4', np.nan],
            'categorical_column_2': ['type_1', 'type_2', np.nan, 'type_4', np.nan],
            'numerical_column_normal_median_1': [0, np.nan, 5, 3, np.nan],
            'numerical_column_normal_median_2': [np.nan, 3, 4, np.nan, 7],
            'categorical_column_cluster_median': ['1', '2', '1', '2', '3'],
            'numerical_column_cluster_median': [4, 11, np.nan, np.nan, np.nan],
        })
        try:
            self.data_handler.fill_missing_values_testing_data(df_test)
        except Exception as error:
            self.assertEqual(error.args[0], 'No missing data parameter was learnt yet')

    def test_fill_missing_values_testing_data__raise_column_error(self):
        df_test = pd.DataFrame({
            'categorical_column_1': ['type_1', np.nan, 'type_3', 'type_4', np.nan],
            'categorical_column_2': ['type_1', 'type_2', np.nan, 'type_4', np.nan],
            'numerical_column_normal_median_1': [0, np.nan, 5, 3, np.nan],
            'numerical_column_normal_median_2': [np.nan, 3, 4, np.nan, 7],
            'numerical_column_cluster_median': [4, 11, np.nan, np.nan, np.nan],
        })
        try:
            self.data_handler.fill_missing_values_training_data(cluster_columns=['categorical_column_cluster_median'])
            self.data_handler.fill_missing_values_testing_data(df_test)
        except Exception as error:
            self.assertEqual(
                error.args[0],
                'One of the cluster columns is missing in the test data: categorical_column_cluster_median'
            )

    def test_fill_missing_values_testing_data__expected_execution(self):
        df_test = pd.DataFrame({
            'categorical_column_1': ['type_1', np.nan, 'type_3', 'type_4', np.nan],
            'categorical_column_2': ['type_1', 'type_2', np.nan, 'type_4', np.nan],
            'numerical_column_normal_median_1': [0, np.nan, 5, 3, np.nan],
            'numerical_column_normal_median_2': [np.nan, 3, 4, np.nan, 7],
            'categorical_column_cluster_median': ['1', '2', '1', '2', '3'],
            'numerical_column_cluster_median': [4, 11, np.nan, np.nan, np.nan],
        })
        self.data_handler.fill_missing_values_training_data(cluster_columns=['categorical_column_cluster_median'])
        filled_test_data_returned = self.data_handler.fill_missing_values_testing_data(df_test)
        filled_test_data_expected = pd.DataFrame({
            'categorical_column_1': ['type_1', 'not informed', 'type_3', 'type_4', 'not informed'],
            'categorical_column_2': ['type_1', 'type_2', 'not informed', 'type_4', 'not informed'],
            'numerical_column_normal_median_1': [0., 3., 5, 3, 3.],
            'numerical_column_normal_median_2': [8., 3, 4, 8., 7],
            'categorical_column_cluster_median': ['1', '2', '1', '2', '3'],
            'numerical_column_cluster_median': [4, 11, 3., 11., 0.],
        })
        pd.testing.assert_frame_equal(filled_test_data_expected,filled_test_data_returned)