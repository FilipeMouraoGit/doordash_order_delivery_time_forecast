import logging
from typing import List
import pandas as pd

logging.basicConfig(level=logging.INFO)


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
