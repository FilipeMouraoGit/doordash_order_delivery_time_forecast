import logging
from typing import List
import pandas as pd
import numpy as np
from doordash_challenge.functions.data_processing.utils import *
logging.basicConfig(level=logging.INFO)

ALLOWED_METRICS = {
    REVENUE_METRIC: [SUBTOTAL_COLUMN, 'sum'],
    N_TRANSACTIONS_METRIC: [TRANSACTIONS_COLUMN, 'sum'],
    N_ITEMS_METRIC: [ITEMS_COLUMN, 'sum']
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
    def get_market_id_kpis(data):
        """
        Given the `data` df group the data with the columns in the `group_columns` list calculating the metrics
        passed in the metrics dictionary with the logic {column_name:['column_to_be_calculated','operation']} ex:
        {'total_spend':['subtotal','sum'], 'n_transactions':['created_at','count'], 'total_items':['items','sum']}
        """
        kpis_dict = {
            TOTAL_REVENUE: f"{np.round(data[SUBTOTAL_COLUMN].sum(),1):,}",
            TOTAL_TRANSACTIONS: f"{data[TRANSACTIONS_COLUMN].sum():,}",
            TOTAL_ITEMS: f"{data[ITEMS_COLUMN].sum():,}",
            DISTINCT_FOOD_CATEGORIES: f"{len(data[STORE_CATEGORY].unique()):,}",
            TOTAL_NUMBER_OF_STORES: f"{len(data[STORE_COLUMN].unique()):,}",
            AVG_NUMBER_OF_ITEMS: f"{(np.round(data[ITEMS_COLUMN].sum() / data[TRANSACTIONS_COLUMN].sum(), 2)):,}",
            AVG_REVENUE: f"{(np.round(data[SUBTOTAL_COLUMN].sum() / data[TRANSACTIONS_COLUMN].sum(), 2)):,}",

        }
        avg_transac = data.groupby(STORE_COLUMN).agg({TRANSACTIONS_COLUMN: 'sum', DAY_COLUMN: 'nunique'})
        avg_transac['avg_daily_transaction'] = avg_transac[TRANSACTIONS_COLUMN]/avg_transac[DAY_COLUMN]
        kpis_dict[P_25_AVG_DAILY_TRANSACTION] = f"{np.round(avg_transac['avg_daily_transaction'].quantile(0.25), 1):,}"
        kpis_dict[P_75_AVG_DAILY_TRANSACTION] = f"{np.round(avg_transac['avg_daily_transaction'].quantile(0.75), 1):,}"
        kpis_dict[P_95_AVG_DAILY_TRANSACTION] = f"{np.round(avg_transac['avg_daily_transaction'].quantile(0.95), 1):,}"
        return kpis_dict

    @staticmethod
    def generate_cumulative_time_series(data, date_column, metric='revenue'):
        grouped_data = DataTransformer.validate_metric_and_group(data, date_column, metric)
        grouped_data = grouped_data.sort_values(date_column)
        grouped_data['cum_metric'] = grouped_data['metric'].cumsum()
        return grouped_data

    @staticmethod
    def generate_percentage_group(data, column, metric='revenue'):
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
