import logging
from typing import List

import pandas as pd

MARKET_ID_COLUMN = 'market_id'
DAY_COLUMN = 'day'
STORE_CATEGORY = 'store_primary_category'
STORE_COLUMN = 'store_id'
DATE_COLUMN = 'created_at'
DELIVERY_COLUMN = 'actual_delivery_time'
STORE_COLUMNS = 'store_id'
ORDER_PLACE_DURATION_COLUMN = 'estimated_order_place_duration'
STORE_CLIENT_DURATION_COLUMN = 'estimated_store_to_consumer_driving_duration'
SUBTOTAL_COLUMN = 'subtotal'
TOTAL_DASHERS_COLUMN = 'total_onshift_dashers'
BUSY_DASHERS_COLUMN = 'total_busy_dashers'
AVAILABLE_DASHERS_COLUMN = 'availabe_dashers_onshift'
TOTAL_ORDERS = 'total_outstanding_orders'
WEEK_COLUMN = 'week'
WEEKDAY_COLUMN = 'weekday'
WEEKEND_COLUMN = 'weekend'
HOUR_COLUMN = 'hour'
TIME_OF_DAY_COLUMN = 'time_of_day'
ESTIMATED_DELIVERY_COLUMN = 'estimated_delivery_time'
DELIVERY_TIME_SECONDS_COLUMN = 'delivery_time'
DELIVERY_TIME_HOURS_COLUMN = 'delivery_time_hours'

logging.basicConfig(level=logging.INFO)


class DataCleaner:
    @staticmethod
    def add_temporal_variables(data: pd.DataFrame):
        """
        From the date column:
         - Extract the week, weekday, hour and a boolean if it is a weekend or not;
         - Based on the hour value estimate the period of the day when the order request was made;
        """
        # temporal variables
        data[WEEK_COLUMN] = data[DATE_COLUMN].dt.isocalendar().week.astype(int)
        data[WEEKDAY_COLUMN] = data[DATE_COLUMN].dt.day_name()
        data[WEEKEND_COLUMN] = (data[DATE_COLUMN].dt.weekday > 4).astype(int)
        data[HOUR_COLUMN] = data[DATE_COLUMN].dt.hour
        data[DAY_COLUMN] = data[DATE_COLUMN].dt.day

        def categorize_time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:  # Above 21 or below 5
                return 'Night'

        data[TIME_OF_DAY_COLUMN] = data.apply(lambda row: categorize_time_of_day(row['hour']), axis=1)
        logging.info(f'''\nDate variables added.\nA total of {data.shape[0]} rows were processed.''')
        return data

    @staticmethod
    def add_target_variables(data: pd.DataFrame, delivery_threshold=3):
        """
        From the date, delivery, estimated order preparation time, and estimated delivery time column:
            - Calculate the delivery time estimation from the other models;
            - Calculate the real delivery time in seconds and hours;
            - Remove the rows with delivery time greater than the allowed threshold;
            - Return the filtered df
        """
        data[ESTIMATED_DELIVERY_COLUMN] = data[ORDER_PLACE_DURATION_COLUMN] + data[STORE_CLIENT_DURATION_COLUMN]
        data[DELIVERY_TIME_SECONDS_COLUMN] = (data[DELIVERY_COLUMN] - data[DATE_COLUMN]) / pd.Timedelta(seconds=1)
        data[DELIVERY_TIME_HOURS_COLUMN] = data[DELIVERY_TIME_SECONDS_COLUMN] / 3600.0
        data_filtered = data.loc[data[DELIVERY_TIME_HOURS_COLUMN].le(delivery_threshold)]
        logging.info(
            f'''\nTarget variables added.\nA total of {data.shape[0]} rows were processed and ''' 
            f'''{data.shape[0] - data_filtered.shape[0]} rows removed due to delivery above the {delivery_threshold}'''
            f''' hours threshold.'''
        )
        return data_filtered

    @staticmethod
    def remove_negative_values(data: pd.DataFrame):
        """
        From the total dashers, busy dashers, subtotal of the order and total outstanding orders columns,:
            - Remove all the rows with negative values for total dashers, subtotal and total orders outstanding;
            - Calculate the available dashers in the region;
            - Clip the available dashers to 0 (do not allow negative values) and remove the busy dashers columns;
        """
        numeric_columns = [SUBTOTAL_COLUMN, TOTAL_DASHERS_COLUMN, BUSY_DASHERS_COLUMN, TOTAL_ORDERS]
        data_filtered = data.copy()
        data_filtered = data_filtered.loc[~(data_filtered[numeric_columns] < 0).any(axis=1)]
        data_filtered[AVAILABLE_DASHERS_COLUMN] = \
            (data_filtered[TOTAL_DASHERS_COLUMN] - data_filtered[BUSY_DASHERS_COLUMN]).clip(lower=0)
        data_filtered = data_filtered.drop(BUSY_DASHERS_COLUMN, axis=1)
        logging.info(
            f'''\nNegative values removed.\nA total of {data.shape[0]} rows were processed and '''
            f'''{data.shape[0] - data_filtered.shape[0]} rows removed due to not allowed negative values for columns'''
            f''' ({SUBTOTAL_COLUMN}, {TOTAL_DASHERS_COLUMN}, {BUSY_DASHERS_COLUMN}, {BUSY_DASHERS_COLUMN})'''
        )
        return data_filtered

    @staticmethod
    def clean_conflict_category(data: pd.DataFrame, columns_to_be_corrected: List=[MARKET_ID_COLUMN, STORE_CATEGORY]):

        """
        Correct the store_id attributes on the list `columns_to_be_corrected` based on 2 rules
        """
        data_corrected = data.copy()
        log_info_string = f'''\nStore data was curated.\nA total of {data.shape[0]} rows were processed\n'''
        for column in columns_to_be_corrected:
            group_data = data \
                .groupby([column, STORE_COLUMN], as_index=False) \
                .agg(first_date=(DATE_COLUMN, 'min'), n_rows=(DATE_COLUMN, 'count'))
            group_data_max_count = group_data.groupby(STORE_COLUMN, as_index=False).agg({'n_rows': 'max'})
            group_data_filtered = group_data.merge(group_data_max_count, how='inner', on=[STORE_COLUMN, 'n_rows'])
            group_data_min_date = group_data_filtered.groupby(STORE_COLUMN, as_index=False).agg({'first_date': 'min'})
            group_data_unique = \
                group_data_filtered.merge(group_data_min_date, how='inner', on=[STORE_COLUMN, 'first_date'])
            data_corrected = data_corrected.drop(column, axis=1)
            data_corrected = \
                data_corrected.merge(group_data_unique[[STORE_COLUMN, column]], how='left', on=[STORE_COLUMN])
            different_elements = (data_corrected[column].to_numpy() != data[column].to_numpy()).sum()
            log_info_string += f'''- For the column {column}, {different_elements} rows were updated\n'''
        logging.info(log_info_string)
        return data_corrected