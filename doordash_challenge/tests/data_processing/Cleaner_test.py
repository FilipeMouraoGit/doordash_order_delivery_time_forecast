import unittest
import pandas as pd
from doordash_challenge.functions.data_processing.Cleaner import DataCleaner
from doordash_challenge.functions.data_processing.Cleaner import MARKET_ID_COLUMN, STORE_CATEGORY, STORE_COLUMN, \
    DATE_COLUMN, DELIVERY_COLUMN, ORDER_PLACE_DURATION_COLUMN, STORE_CLIENT_DURATION_COLUMN, \
    SUBTOTAL_COLUMN, TOTAL_DASHERS_COLUMN, BUSY_DASHERS_COLUMN, AVAILABLE_DASHERS_COLUMN, TOTAL_ORDERS


class DataCleanerTest(unittest.TestCase):

    def test_add_temporal_variables(self):
        df_data = pd.DataFrame({DATE_COLUMN: [
            '2024-10-23 05:35:11', '2024-11-01 04:22:01', '2024-11-05 11:25:14', '2024-11-07 15:31:57',
            '2024-11-11 17:45:47', '2024-11-13 19:45:41', '2024-11-15 21:35:11', '2024-11-16 01:27:11']})
        df_data[DATE_COLUMN] = pd.to_datetime(df_data[DATE_COLUMN])
        df_expected = pd.DataFrame({
            DATE_COLUMN: [
                '2024-10-23 05:35:11', '2024-11-01 04:22:01', '2024-11-05 11:25:14', '2024-11-07 15:31:57',
                '2024-11-11 17:45:47', '2024-11-13 19:45:41', '2024-11-15 21:35:11', '2024-11-16 01:27:11'],
            'week': [43, 44, 45, 45, 46, 46, 46, 46],
            'weekday': ['Wednesday', 'Friday', 'Tuesday', 'Thursday', 'Monday', 'Wednesday', 'Friday', 'Saturday'],
            'weekend': [0, 0, 0, 0, 0, 0, 0, 1],
            'hour': [5, 4, 11, 15, 17, 19, 21, 1],
            'time_of_day': [
                'Morning', 'Night', 'Morning', 'Afternoon', 'Evening', 'Evening', 'Night', 'Night']
        })
        df_expected[DATE_COLUMN] = pd.to_datetime(df_expected[DATE_COLUMN])
        for column in ['week', 'weekend', 'hour']:
            df_expected[column] = df_expected[column].astype(int)
        df_returned = DataCleaner.add_temporal_variables(df_data)
        df_returned['hour'] = df_returned['hour'].astype(int)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_add_target_variables(self):
        df_data = pd.DataFrame({
            DATE_COLUMN: [
                '2024-10-23 05:35:11', '2024-11-01 04:22:01', '2024-11-05 11:25:14', '2024-11-07 15:31:57',
                '2024-11-11 17:45:47', '2024-11-13 19:45:41', '2024-11-15 21:35:11', '2024-11-16 01:27:11'],
            DELIVERY_COLUMN: [
                '2024-10-23 05:55:11', '2024-11-01 08:22:01', '2024-11-05 11:28:14', '2024-11-07 16:01:57',
                '2024-11-11 19:45:47', '2024-11-13 22:15:41', '2024-11-16 00:35:11', '2024-11-16 04:27:12'],
            ORDER_PLACE_DURATION_COLUMN: [600, 500, 100, 500, 1200, 700, 100, 200],
            STORE_CLIENT_DURATION_COLUMN: [300, 1000, 400, 600, 1500, 900, 50, 300]
        })
        df_data[DATE_COLUMN] = pd.to_datetime(df_data[DATE_COLUMN])
        df_data[DELIVERY_COLUMN] = pd.to_datetime(df_data[DELIVERY_COLUMN])

        df_expected = pd.DataFrame({
            DATE_COLUMN: [
                '2024-10-23 05:35:11',  '2024-11-05 11:25:14', '2024-11-07 15:31:57', '2024-11-11 17:45:47',
                '2024-11-13 19:45:41', '2024-11-15 21:35:11'],
            DELIVERY_COLUMN: [
                '2024-10-23 05:55:11',  '2024-11-05 11:28:14', '2024-11-07 16:01:57', '2024-11-11 19:45:47',
                '2024-11-13 22:15:41', '2024-11-16 00:35:11'],
            ORDER_PLACE_DURATION_COLUMN: [600, 100, 500, 1200, 700, 100],
            STORE_CLIENT_DURATION_COLUMN: [300, 400, 600, 1500, 900, 50],
            'estimated_delivery_time': [900, 500, 1100, 2700, 1600, 150],
            'delivery_time': [20*60, 3*60, 30*60, 2*60*60, 2.5*60*60, 3*60*60],
            'delivery_time_hours': [1/3, 1/20, 0.5, 2, 2.5, 3],
        })
        df_expected[DATE_COLUMN] = pd.to_datetime(df_expected[DATE_COLUMN])
        df_expected[DELIVERY_COLUMN] = pd.to_datetime(df_expected[DELIVERY_COLUMN])
        df_returned = DataCleaner.add_target_variables(df_data)
        df_returned = df_returned.reset_index(drop=True)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_remove_negative_values(self):
        df_data = pd.DataFrame({
            TOTAL_DASHERS_COLUMN: [1, -1,  3, 6, 9, 4, 4, 2, 2, 20],
            BUSY_DASHERS_COLUMN:  [1,  2, -1, 4, 5, 5, 9, 9, 1, 10],
            SUBTOTAL_COLUMN: [1, 2, 3, -1, 5, 6, 7, 8, 9, 10],
            TOTAL_ORDERS: [1, 2, 3, 4, -1, 6, 7, 8, 9, 10]
        })
        df_expected = pd.DataFrame({
            TOTAL_DASHERS_COLUMN: [1, 4, 4, 2, 2, 20],
            SUBTOTAL_COLUMN: [1, 6, 7, 8, 9, 10],
            TOTAL_ORDERS: [1, 6, 7, 8, 9, 10],
            AVAILABLE_DASHERS_COLUMN: [0, 0, 0, 0, 1, 10]
        })
        df_returned = DataCleaner.remove_negative_values(df_data)
        df_returned = df_returned.reset_index(drop=True)
        pd.testing.assert_frame_equal(df_expected, df_returned)

    def test_clean_conflict_category(self):
        df_data = pd.DataFrame({
            STORE_COLUMN: ['1', '1', '1', '1', '1', '2', '2', '2', '3', '3', '3'],
            MARKET_ID_COLUMN:  ['1', '1', '1', '2', '2', '1', '2', '2', '1', '2', '3'],
            STORE_CATEGORY: ['a', 'b', 'c', 'd', 'a', 'a', 'b', 'c', 'a', 'b', 'd'],
            DATE_COLUMN: [
                '2024-10-23', '2024-11-01', '2024-11-05', '2024-11-07',  '2024-11-07',
                '2024-11-11', '2024-11-13', '2024-11-15', '2024-11-16', '2024-11-17', '2024-10-18']
        })
        df_data[DATE_COLUMN] = pd.to_datetime(df_data[DATE_COLUMN])
        df_expected = pd.DataFrame({
            STORE_COLUMN: ['1', '1', '1', '1', '1', '2', '2', '2', '3', '3', '3'],
            DATE_COLUMN: [
                '2024-10-23', '2024-11-01', '2024-11-05', '2024-11-07', '2024-11-07',
                '2024-11-11', '2024-11-13', '2024-11-15', '2024-11-16', '2024-11-17', '2024-10-18'],
            MARKET_ID_COLUMN:  ['1', '1', '1', '1', '1', '2', '2', '2', '3', '3', '3'],
            STORE_CATEGORY: ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'd', 'd', 'd']

        })
        df_expected[DATE_COLUMN] = pd.to_datetime(df_expected[DATE_COLUMN])
        df_returned = DataCleaner.clean_conflict_category(df_data)
        df_returned = df_returned.reset_index(drop=True)
        pd.testing.assert_frame_equal(df_expected, df_returned)