import unittest
import pandas as pd
from doordash_challenge.functions.data_processing.Cleaner import DataCleaner
from doordash_challenge.functions.data_processing.Cleaner import STORE_COLUMN, \
    DATE_COLUMN, DELIVERY_COLUMN, STORE_COLUMNS, ORDER_PLACE_DURATION_COLUMN, STORE_CLIENT_DURATION_COLUMN, \
    SUBTOTAL_COLUMN, TOTAL_DASHERS_COLUMN, BUSY_DASHERS_COLUMN, AVAILABLE_DASHERS_COLUMN, TOTAL_ORDERS


class FirstTest(unittest.TestCase):

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
        pd.testing.assert_frame_equal(df_expected, df_returned)