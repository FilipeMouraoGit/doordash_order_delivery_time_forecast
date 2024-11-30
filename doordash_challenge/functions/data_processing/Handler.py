import logging
from typing import List
import pandas as pd
from doordash_challenge.functions.data_processing.utils import *
import numpy as np

logging.basicConfig(level=logging.INFO)
## add new test for nan category value

class DataHandler:
    def __init__(self, data: pd.DataFrame, fill_column_methods: dict, percentile_description_dict: dict):
        """
        Initialize the class with a df with columns that can have cells with nan values.
        The dictionary of methods to be used defines which method is going to be used for each column, ex:
        {'column_1': 'fill with new category', 'column_2': 'fill with median', 'column_3': 'fill with cluster median'}
        The percentile dict defines the numerical columns used to create the percentile categories columns ex:
        {'column_3:[0.2, 0.5, 0.8]} would calculate percentiles 20, 50 and 80 for column_3
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
        self.percentile_description_dict = percentile_description_dict
        self.categories_percentile_dict = {}

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
            logging.info(f'''\n{na_num} rows filled with value `{value}` for column {column}''')

        else:
            for method_column, method in self.fill_column_methods.items():
                if method == FILL_NA_WITH_NEW_CATEGORY_METHOD:
                    self.fill_with_value[method_column] = extra_category_name
                    self.handler_data[method_column] = self.raw_data[method_column].fillna(extra_category_name)
                    self.handler_data[method_column] = self.handler_data[method_column].astype(str)
                    na_num, value = self.raw_data[method_column].isna().sum(), extra_category_name
                    logging.info(f'''\n{na_num} rows filled with value `{value}` for column {method_column}''')

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
            logging.info(f'''\n{na_num} rows filled with value `{value}` for column {column}''')
        else:
            for method_column, method in self.fill_column_methods.items():
                if method == FILL_NA_WITH_MEDIAN_METHOD:
                    median = self.raw_data[method_column].median()
                    self.fill_with_value[method_column] = median
                    self.handler_data[method_column] = self.raw_data[method_column].fillna(median)
                    na_num, value = self.raw_data[method_column].isna().sum(), median
                    logging.info(f'''\n{na_num} rows filled with value `{value}` for column {method_column}''')
        return self.handler_data

    def _generate_column_cluster_data(self, raw_cluster_data, cluster_columns, column):
        for cluster_column in cluster_columns:
            raw_cluster_data[cluster_column] = raw_cluster_data[cluster_column].fillna('non_value')
        cluster_data = raw_cluster_data.groupby(cluster_columns, as_index=False).agg(median=(column, 'median'))
        cluster_median = raw_cluster_data.merge(cluster_data, on=cluster_columns, how='left')
        cluster_median['column_with_cluster_median'] = cluster_median[column].fillna(cluster_median['median'])
        return cluster_data, cluster_median['column_with_cluster_median'].to_list()

    def fill_na_with_cluster_median(self, cluster_columns: List):
        """
        Split the data in different groups based on the list of cluster columns and fill each group with the
        specific median
        """
        for column, method in self.fill_column_methods.items():
            if method == FILL_NA_WITH_CLUSTER_MEDIAN_METHOD:
                raw_cluster_data = self.raw_data.copy()
                cluster_data, cluster_median_values = \
                    self._generate_column_cluster_data(raw_cluster_data, cluster_columns, column)
                self.fill_with_cluster_value[column] = cluster_data
                self.handler_data[column] = cluster_median_values
                na_num = self.raw_data[column].isna().sum()
                logging.info(
                    f'''\n{na_num} rows for column {column} filled with cluster df:\n {cluster_data.to_string()}'''
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
        test_data_no_na = test_data.copy()
        if len(self.fill_with_value) == 0 and len(self.fill_with_cluster_value) == 0:
            raise ValueError('No missing data parameter was learnt yet')
        for column, method in self.fill_column_methods.items():
            if method in [FILL_NA_WITH_NEW_CATEGORY_METHOD, FILL_NA_WITH_MEDIAN_METHOD]:
                test_data_no_na[column] = test_data[column].fillna(self.fill_with_value[column])
            elif method == FILL_NA_WITH_CLUSTER_MEDIAN_METHOD:
                cluster_data = self.fill_with_cluster_value[column]
                cluster_columns = list(cluster_data.columns)[:-1]
                test_data_raw = test_data.copy()
                for cluster_column in cluster_columns:
                    if cluster_column not in list(test_data.columns):
                        raise ValueError(f'One of the cluster columns is missing in the test data: {cluster_column}')
                    test_data_raw[cluster_column] = test_data_raw[cluster_column].fillna('non_value')
                data = test_data_raw.merge(cluster_data, on=cluster_columns, how='left')
                data['cluster_median'] = data[column].fillna(data['median'])
                test_data_no_na[column] = data['cluster_median'].to_list()
        return test_data_no_na

    def _generate_percentile_values(self, percentile_data):
        """
        Calculate the percentile_category_values_dict values categories and values for each column defined in the
        percentile_description_dict, return the results in the percentile_category_values_dict
            From the percentile_description_dict:
            - Calculate the percentiles based on the percentile_data df;
            - Define the percentile categories based on the values;
            - Save the results in the percentile_category_values_dict
        """
        percentile_category_values_dict = {}
        for column, percentiles in self.percentile_description_dict.items():
            initial_quantile = np.round(percentile_data[column].quantile(percentiles[0]), 2)
            category_dict = {f'Below Q{int(percentiles[0] * 100)}': initial_quantile}
            for i in range(1, len(percentiles)):
                val = percentiles[i]
                old_val = percentiles[i - 1]
                category_dict[f'Between Q{int(old_val * 100)} and Q{int(val * 100)}'] = np.round(
                    percentile_data[column].quantile(val), 2)
            last_quantile = np.round(percentile_data[column].quantile(percentiles[-1]), 2)
            category_dict[f'Above Q{int(percentiles[-1] * 100)}'] = last_quantile
            percentile_category_values_dict[column] = category_dict
        return percentile_category_values_dict

    def generate_percentile_categories(self, percentile_data, key_column=STORE_COLUMN):
        """
        Generate the dictionary mappers, the key_column is a categorical column that would produce a
        sparse dataset if a one hot encoding method would be applied, it is going to be converted in multiple
        percentile columns using the percentile_data and the percentile_description_dict used when initializing the
        class
        """
        percentile_category_values_dict = self._generate_percentile_values(percentile_data)

        def selecting_category(price, percentiles):
            for key, value in percentiles.items():
                if price <= value:
                    return key
            return list(percentiles.keys())[-1]

        for column, percentiles in percentile_category_values_dict.items():
            perc_df = percentile_data[[key_column, column]].copy()
            perc_df['category'] = perc_df.apply(lambda row: selecting_category(row[column], percentiles), axis=1)
            perc_dict = perc_df.set_index(key_column)['category'].to_dict()
            self.categories_percentile_dict[f'store_perc_category_{column}'] = perc_dict
            self.handler_data[f'store_perc_category_{column}'] = self.raw_data[key_column].map(perc_dict)
        return self.handler_data

    def map_percentile_categories_unseen_data(self, unseen_data, key_column=STORE_COLUMN):
        """
        Use the learning categories_percentile_dict, apply the categories learnt from the percentile_data into the
        unseen data
        """
        for column, mapper in self.categories_percentile_dict.items():
            unseen_data[column] = unseen_data[key_column].map(mapper)
        return unseen_data
