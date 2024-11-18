import unittest
import pandas as pd
from doordash_challenge.functions.data_processing.Transformer import DataTransformer


class TransformerTest(unittest.TestCase):

    def test_group_data__single_column_single_metric(self):
        df_data = pd.DataFrame({
            'group_column_1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'group_column_2': ['1', '1', '2', '2', '3', '3', '4', '4'],
            'boolean_column': [0, 1, 1, 0, 0, 0, 1, 1],
            'total_column': [100, 150, 200, 350, 30, 40, 50, 70],
            'items_column': [5, 6, 3, 4, 1, 8, 3, 6]
        })
        df_expected = pd.DataFrame({'group_column_1': ['A', 'B'], 'total_sales': [800, 190]})
        agg_methods = {'total_sales': ('total_column', 'sum')}
        df_returned = DataTransformer.group_data(df_data, group_columns=['group_column_1'], agg_methods=agg_methods)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_group_data__double_column_single_metric(self):
        df_data = pd.DataFrame({
            'group_column_1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'group_column_2': ['1', '1', '2', '2', '3', '3', '4', '4'],
            'boolean_column': [0, 1, 1, 0, 0, 0, 1, 1],
            'total_column': [100, 150, 200, 350, 30, 40, 50, 70],
            'items_column': [5, 6, 3, 4, 1, 8, 3, 6]
        })
        df_expected = pd.DataFrame({
            'group_column_1': ['A', 'A', 'B', 'B'],
            'group_column_2': ['1', '2', '3', '4'],
            'total_sales': [250, 550, 70, 120]
        })
        agg_methods = {'total_sales': ('total_column', 'sum')}
        df_returned = DataTransformer\
            .group_data(df_data, group_columns=['group_column_1', 'group_column_2'], agg_methods=agg_methods)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_group_data__double_column_double_metric(self):
        df_data = pd.DataFrame({
            'group_column_1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'group_column_2': ['1', '1', '2', '2', '3', '3', '4', '4'],
            'boolean_column': [0, 1, 1, 0, 0, 0, 1, 1],
            'total_column': [100, 150, 200, 350, 30, 40, 50, 70],
            'items_column': [5, 6, 3, 4, 1, 8, 3, 6]
        })
        df_expected = pd.DataFrame({
            'group_column_1': ['A', 'A', 'B', 'B'],
            'group_column_2': ['1', '2', '3', '4'],
            'total_sales': [250, 550, 70, 120],
            'n_sales': [2, 2, 2, 2]
        })
        agg_methods = {'total_sales': ('total_column', 'sum'), 'n_sales': ('total_column', 'count')}
        df_returned = DataTransformer\
            .group_data(df_data, group_columns=['group_column_1', 'group_column_2'], agg_methods=agg_methods)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_group_data__double_column_multiple_metrics(self):
        df_data = pd.DataFrame({
            'group_column_1': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'group_column_2': ['1', '1', '2', '2', '3', '3', '4', '4'],
            'boolean_column': [0, 1, 1, 0, 0, 0, 1, 1],
            'total_column': [100, 150, 200, 350, 30, 40, 50, 70],
            'items_column': [5, 6, 3, 4, 1, 8, 3, 6]
        })
        df_expected = pd.DataFrame({
            'group_column_1': ['A', 'A', 'B', 'B'],
            'group_column_2': ['1', '2', '3', '4'],
            'total_sales': [250, 550, 70, 120],
            'n_sales': [2, 2, 2, 2],
            'median_booleans': [0.5, 0.5, 0., 1.],
            'avg_items': [5.5, 3.5, 4.5, 4.5]
        })
        agg_methods = {
            'total_sales': ('total_column', 'sum'),
            'n_sales': ('total_column', 'count'),
            'median_booleans': ('boolean_column', 'median'),
            'avg_items': ('items_column', 'mean')
        }
        df_returned = DataTransformer\
            .group_data(df_data, group_columns=['group_column_1', 'group_column_2'], agg_methods=agg_methods)
        pd.testing.assert_frame_equal(df_expected, df_returned)