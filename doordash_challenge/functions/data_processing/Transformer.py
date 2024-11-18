import logging
from typing import List
import pandas as pd
import numpy as np
logging.basicConfig(level=logging.INFO)

ALLOWED_METRICS = {
    'number of transactions': ['transactions', 'sum'],
    'revenue': ['subtotal', 'sum'],
    'number of items': ['items', 'sum']
}
class DataTransformer:
    @staticmethod
    def group_data(data: pd.DataFrame, group_columns: List, agg_methods: dict):
        """
        Given the `data` df group the data with the columns in the `group_columns` list calculating the metrics
        passed in the metrics dictionary with the logic {column_name:['column_to_be_calculated','operation']} ex:
        {'total_spend':['subtotal','sum'], 'n_transactions':['created_at','count'], 'total_items':['items','sum']}
        """
        agg_dict = {key: (value[0], value[1]) for key, value in agg_methods.items()}
        grouped_data = data.groupby(group_columns, as_index=False).agg(**agg_dict)
        return grouped_data

    @staticmethod
    def validate_metric_and_group(data, column, metric):
        if metric not in list(ALLOWED_METRICS.keys()):
            raise ValueError(f'Metric `{metric}` not supported')
        grouped_data = DataTransformer.group_data(
            data, group_columns=[column], agg_methods={'metric': ALLOWED_METRICS[metric]}
        )
        return grouped_data
    @staticmethod
    def get_column_totals(data, total_columns: List):
        """
        Given the `data` df group the data with the columns in the `group_columns` list calculating the metrics
        passed in the metrics dictionary with the logic {column_name:['column_to_be_calculated','operation']} ex:
        {'total_spend':['subtotal','sum'], 'n_transactions':['created_at','count'], 'total_items':['items','sum']}
        """
        total_dict = {}
        for column in total_columns:
            total_dict[column] = data[column].sum()
        return total_dict

    @staticmethod
    def generate_cumulative_time_series(data, date_column, metric='revenue'):
        grouped_data = DataTransformer.validate_metric_and_group(data, date_column, metric)
        grouped_data = grouped_data.sort_values(date_column)
        grouped_data['cum_metric'] = grouped_data['metric'].cumsum()
        return grouped_data
    @staticmethod
    def generate_percentage_grop(data, column, metric='revenue'):
        grouped_data = DataTransformer.validate_metric_and_group(data, column, metric)
        grouped_data = grouped_data.set_index(column)
        percentage_df = 100 * np.round(grouped_data / grouped_data.sum(), 2)
        percentage_df = percentage_df.reset_index()
        return percentage_df

    @staticmethod
    def generate_rank_group(data, column, metric='revenue', rank=5):
        grouped_data = DataTransformer.validate_metric_and_group(data, column, metric)
        grouped_data = grouped_data.sort_values('metric')
        top_data = grouped_data.tail(rank)
        return top_data

