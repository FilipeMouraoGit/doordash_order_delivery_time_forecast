import logging
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO)
FILL_NA_WITH_NEW_CATEGORY_METHOD = 'fill with new category'
FILL_NA_WITH_MEDIAN_METHOD = 'fill with median'
FILL_NA_WITH_CLUSTER_MEDIAN_METHOD = 'fill with cluster median'


class DataHandler:
    def __init__(self, data: pd.DataFrame, fill_column_methods: dict):
        """
        Initialize the class with  a dataframe with columns that have rows with nan values, the dictionary
        of methods to be used defines which method is going to be used for each column, ex:
        {'column_1': 'fill with new category', 'column_2': 'fill with median', 'column_3': 'fill with cluster median'}
        """
        self.raw_data = data.copy()
        self.handler_data = data.copy()
        self.fill_with_value = {}
        self.fill_with_cluster_value = {}
        for method in set(fill_column_methods.values()):
            allowed_methods = \
                [FILL_NA_WITH_NEW_CATEGORY_METHOD, FILL_NA_WITH_MEDIAN_METHOD, FILL_NA_WITH_CLUSTER_MEDIAN_METHOD]
            if method not in allowed_methods:
                raise ValueError(f'Method `{method}` not supported by the Handler')
            else:
                self.fill_column_methods = fill_column_methods

    def fill_na_with_new_category(self, extra_category_name='not informed', column=None):
        """
        Add a new value for the categorical columns with nan values passed in the columns methods, or
        a specific column passed as a parameter
        """
        if column is not None and self.fill_column_methods[column] == FILL_NA_WITH_NEW_CATEGORY_METHOD:
            self.fill_with_value[column] = extra_category_name
            self.handler_data[column] = self.raw_data[column].fillna(extra_category_name)
            self.handler_data[column] = self.handler_data[column].astype(str)
            na_num, value = self.raw_data[column].isna().sum(), extra_category_name
            logging.info(f'''{na_num} rows filled with value `{value}` for column {column}''')

        else:
            for method_column, method in self.fill_column_methods.items():
                if method == FILL_NA_WITH_NEW_CATEGORY_METHOD:
                    self.fill_with_value[method_column] = extra_category_name
                    self.handler_data[method_column] = self.raw_data[method_column].fillna(extra_category_name)
                    self.handler_data[method_column] = self.handler_data[method_column].astype(str)
                    na_num, value = self.raw_data[method_column].isna().sum(), extra_category_name
                    logging.info(f'''{na_num} rows filled with value `{value}` for column {method_column}''')

        return self.handler_data

    def fill_na_with_median(self, column=None):
        """
        Use the median to fill numerical columns with nan values passed in the columns methods, or
        a specific column passed as a parameter
        """
        if column is not None and self.fill_column_methods[column] == FILL_NA_WITH_MEDIAN_METHOD:
            median = self.raw_data[column].median()
            self.fill_with_value[column] = median
            self.handler_data[column] = self.raw_data[column].fillna(median)
            na_num, value = self.raw_data[column].isna().sum(), median
            logging.info(f'''{na_num} rows filled with value `{value}` for column {column}''')
        else:
            for method_column, method in self.fill_column_methods.items():
                if method == FILL_NA_WITH_MEDIAN_METHOD:
                    median = self.raw_data[method_column].median()
                    self.fill_with_value[method_column] = median
                    self.handler_data[method_column] = self.raw_data[method_column].fillna(median)
                    na_num, value = self.raw_data[method_column].isna().sum(), median
                    logging.info(f'''{na_num} rows filled with value `{value}` for column {method_column}''')
        return self.handler_data

    def fill_na_with_cluster_median(self, cluster_columns: List):
        """
        Split the data in different groups based on the list of cluster columns and fill each group with the
        specific median
        """
        for column, method in self.fill_column_methods.items():
            if method == FILL_NA_WITH_CLUSTER_MEDIAN_METHOD:
                cluster_data = self.raw_data \
                    .groupby(cluster_columns, as_index=False) \
                    .agg(median=(column, 'median'))
                self.fill_with_cluster_value[column] = cluster_data
                data = self.raw_data.merge(cluster_data, on=cluster_columns, how='left')
                self.handler_data[column] = data[column].fillna(data['median'])
                na_num = self.raw_data[column].isna().sum()
                logging.info(
                    f'''{na_num} rows for column {column} filled with cluster df:\n {cluster_data.to_string()}'''
                )

        return self.handler_data

    def fill_missing_values_training_data(self, cluster_columns: List):
        """
        Fill the missing data for all columns in the input df with the respective methods described in the
        input dictionary
        """
        _ = self.fill_na_with_new_category()
        _ = self.fill_na_with_median()
        handler_data = self.fill_na_with_cluster_median(cluster_columns)
        return handler_data

    def fill_missing_values_testing_data(self, test_data):
        """
        Fill missing values with the values learnt from the training data
        """
        if len(self.fill_with_value) == 0 and len(self.fill_with_cluster_value) == 0:
            raise ValueError('No missing data parameter was learnt yet')
        for column, method in self.fill_column_methods.items():
            if method in [FILL_NA_WITH_NEW_CATEGORY_METHOD, FILL_NA_WITH_MEDIAN_METHOD]:
                test_data[column] = test_data[column].fillna(self.fill_with_value[column])
            elif method == FILL_NA_WITH_CLUSTER_MEDIAN_METHOD:
                cluster_data = self.fill_with_cluster_value[column]
                cluster_columns = list(cluster_data.columns)[:-1]
                for cluster_column in cluster_columns:
                    if cluster_column not in list(test_data.columns):
                        raise ValueError(f'One of the cluster columns is missing in the test data: {cluster_column}')
                data = test_data.merge(cluster_data, on=cluster_columns, how='left')
                test_data[column] = data[column].fillna(data['median'])
                # If there is a new value never seen in a cluster, fill with 0
                test_data[column] = test_data[column].fillna(0)
        return test_data
