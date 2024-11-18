import logging
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
        self.fill_column_methods = fill_column_methods
        self.fill_with_value = {}
        self.fill_with_cluster_value = {}

    def fill_na_with_new_category(self, extra_category_name, column=None):
        """
        Add a new value for the categorical columns with nan values passed in the columns methods
        """
        if column is not None and self.fill_column_methods[column] == FILL_NA_WITH_NEW_CATEGORY_METHOD:
            self.fill_with_value[column] = extra_category_name
            self.handler_data[column] = self.raw_data[column].fillna(extra_category_name)
            self.handler_data[column] = self.handler_data[column].astype(str)
        else:
            for method_column, method in self.fill_column_methods.items():
                if method == FILL_NA_WITH_NEW_CATEGORY_METHOD:
                    self.fill_with_value[method_column] = extra_category_name
                    self.handler_data[method_column] = self.raw_data[method_column].fillna(extra_category_name)
                    self.handler_data[method_column] = self.handler_data[method_column].astype(str)

        return self.handler_data

