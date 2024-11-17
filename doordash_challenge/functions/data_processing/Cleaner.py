import logging
import pandas as pd

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
logging.basicConfig(level=logging.INFO)


class DataCleaner:
    @staticmethod
    def add_temporal_variables(data: pd.DataFrame):
        """
        From a date column, extract the weekday, hour, time_of_day and weekend
        """
        # temporal variables
        data['week'] = data[DATE_COLUMN].dt.isocalendar().week.astype(int)
        data['weekday'] = data[DATE_COLUMN].dt.day_name()
        data["weekend"] = (data[DATE_COLUMN].dt.weekday > 4).astype(int)
        data['hour'] = data[DATE_COLUMN].dt.hour

        def categorize_time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:  # Above 21 or below 5
                return 'Night'

        data['time_of_day'] = data.apply(lambda row: categorize_time_of_day(row['hour']), axis=1)
        logging.info(f'''Date variables added.\nA total of {data.shape[0]} rows processed.''')
        return data

    @staticmethod
    def add_target_variables(data: pd.DataFrame, delivery_threshold=3):
        """
        From a date column, extract the weekday, hour, time_of_day and weekedn
        """
        data['estimated_delivery_time'] = data[ORDER_PLACE_DURATION_COLUMN] + data[STORE_CLIENT_DURATION_COLUMN]
        data['delivery_time'] = (data[DELIVERY_COLUMN] - data[DATE_COLUMN]) / pd.Timedelta(seconds=1)
        data['delivery_time_hours'] = data['delivery_time'] / 3600.0
        data_filtered = data.loc[data['delivery_time_hours'].le(delivery_threshold)]
        logging.info(
            f'''Target variables added.\nA total of {data.shape[0]} rows processed and ''' 
            f'''{data.shape[0] - data_filtered.shape[0]} rows removed due to delivery above the {delivery_threshold}'''
            f''' hours threshold.'''
        )
        return data_filtered
