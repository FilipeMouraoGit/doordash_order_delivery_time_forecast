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

    def test_generate_cumulative_time_series__default_metric(self):
        df_data = pd.DataFrame({
            'day': ['1', '1', '2', '3', '4', '5', '6', '6', '7', '8'],
            'week': ['1', '1', '1', '2', '2', '2', '3', '3', '3', '3'],
            'subtotal': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'transactions': [5, 6, 7, 2, 1, 1, 4, 3, 4, 7],
            'items': [2, 1, 4, 5, 6, 7, 8, 2, 10, 7]
        })
        df_expected = pd.DataFrame({
            'day': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'metric': [30, 30, 40, 50, 60, 150, 90, 100],
            'cum_metric': [30, 60, 100, 150, 210, 360, 450, 550]
        })

        df_returned = DataTransformer.generate_cumulative_time_series(df_data, date_column='day')
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_generate_cumulative_time_series__default_metric_week(self):
        df_data = pd.DataFrame({
            'day': ['1', '1', '2', '3', '4', '5', '6', '6', '7', '8'],
            'week': ['1', '1', '1', '2', '2', '2', '3', '3', '3', '3'],
            'subtotal': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'transactions': [5, 6, 7, 2, 1, 1, 4, 3, 4, 7],
            'items': [2, 1, 4, 5, 6, 7, 8, 2, 10, 7]
        })
        df_expected = pd.DataFrame({'week': ['1', '2', '3'], 'metric': [60, 150, 340], 'cum_metric': [60, 210, 550]})
        df_returned = DataTransformer.generate_cumulative_time_series(df_data, date_column='week')
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_generate_cumulative_time_series__items_metric(self):
        df_data = pd.DataFrame({
            'day': ['1', '1', '2', '3', '4', '5', '6', '6', '7', '8'],
            'week': ['1', '1', '1', '2', '2', '2', '3', '3', '3', '3'],
            'subtotal': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'transactions': [5, 6, 7, 2, 1, 1, 4, 3, 4, 7],
            'items': [2, 1, 4, 5, 6, 7, 8, 2, 10, 7]
        })
        df_expected = pd.DataFrame({
            'day': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'metric': [3, 4, 5, 6, 7, 10, 10, 7],
            'cum_metric': [3, 7, 12, 18, 25, 35, 45, 52]
        })

        df_returned = DataTransformer\
            .generate_cumulative_time_series(df_data, date_column='day', metric='number of items')
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_generate_cumulative_time_series__transaction_metric(self):
        df_data = pd.DataFrame({
            'day': ['1', '1', '2', '3', '4', '5', '6', '6', '7', '8'],
            'week': ['1', '1', '1', '2', '2', '2', '3', '3', '3', '3'],
            'subtotal': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'transactions': [5, 6, 7, 2, 1, 1, 4, 3, 4, 7],
            'items': [2, 1, 4, 5, 6, 7, 8, 2, 10, 7]
        })
        df_expected = pd.DataFrame({
            'day': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'metric': [11, 7, 2, 1, 1, 7, 4, 7],
            'cum_metric': [11, 18, 20, 21, 22, 29, 33, 40]
        })

        df_returned = DataTransformer\
            .generate_cumulative_time_series(df_data, date_column='day', metric='number of transactions')
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_generate_cumulative_time_series__raise_error(self):
        df_data = pd.DataFrame({
            'day': ['1', '1', '2', '3', '4', '5', '6', '6', '7', '8'],
            'week': ['1', '1', '1', '2', '2', '2', '3', '3', '3', '3'],
            'subtotal': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'transactions': [5, 6, 7, 2, 1, 1, 4, 3, 4, 7],
            'items': [2, 1, 4, 5, 6, 7, 8, 2, 10, 7]
        })

        try:
            _ = DataTransformer \
                .generate_cumulative_time_series(df_data, date_column='day', metric='new metric')
        except Exception as error:
            self.assertEqual(error.args[0], 'Metric `new metric` not supported')