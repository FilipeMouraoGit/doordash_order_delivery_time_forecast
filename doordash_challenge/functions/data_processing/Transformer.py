import logging
from typing import List
import pandas as pd

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
    def generate_cumulative_time_series(data, date_column, metric='revenue'):
        if metric not in list(ALLOWED_METRICS.keys()):
            raise ValueError(f'Metric `{metric}` not supported')
        grouped_data = DataTransformer.group_data(
            data, group_columns=[date_column], agg_methods={'metric': ALLOWED_METRICS[metric]}
        )
        grouped_data = grouped_data.sort_values(date_column)
        grouped_data['cum_metric'] = grouped_data['metric'].cumsum()
        return grouped_data

