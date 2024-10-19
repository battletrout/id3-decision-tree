import pandas as pd
from dataLogger import debug_log
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class DataSplitifyzer:
    """
    Split, Stratify, Standardize the data. Remove unwanted columns at runtime. Called by model functions on init.
    Data should already be pre-processed into a .csv file with headers that can be read by pandas.
    """
    def __init__(self, log_enabled:bool = False):
        self.standardize_mappings = []
        self.log_enabled = log_enabled

    def downcast_data(self,data: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast int and float values to smaller values where possible
        e.g. float64 -> float32, int64 -> int16
        """
        
        if self.log_enabled:
            debug_log(True, "Downcasting dataframe, initial size:")
            data.info(memory_usage="deep")
            print()
        
        data = data.apply(pd.to_numeric, downcast='float')
        data = data.apply(pd.to_numeric, downcast='integer')
        
        if self.log_enabled:
            debug_log(True, "Done. Resulting size:")
            data.info(memory_usage="deep")
            print()
        
        return data

    def standardize(self, train_data: pd.DataFrame, test_data: pd.DataFrame, standardize_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Given a training and test set, performs z-score standardization.
        Args
            train data and test dataframes
        returns
            standardized data and dataframes
        """
        standardization_map = {}
        for column in standardize_columns:
            train_mean = train_data[column].mean()
            train_std = train_data[column].std()
            # retain scaler in the standardize mappings dict
            standardization_map[column] = {'mean' : train_mean, 'std' : train_std}    
            train_data[column] = (train_data[column] - train_mean) / train_std  # = (train_data - train_data.mean(axis=1)) / train_data.std(1)
            test_data[column] = (test_data[column] - train_mean) / train_std
        self.standardize_mappings.append(standardization_map)
        return train_data, test_data

    def standardize_from_mappings(self, validation_data: pd.DataFrame, standardize_columns: List[str], standardize_mappings:dict) -> pd.DataFrame:
        for column in standardize_columns:
            train_mean = standardize_mappings[column]['mean']
            train_std = standardize_mappings[column]['std']        
            validation_data[column] = (validation_data[column] - train_mean) / train_std  # = (train_data - train_data.mean(axis=1)) / train_data.std(1)
        return validation_data

    def split_data(self, data: pd.DataFrame, target_column: str, 
                   cv_type: str = '5x2', stratify: bool = True,
                   standardize_columns: List[str] = [],
                   exclude_columns: List[str] = [], downcast: bool = True) -> Tuple[List[pd.DataFrame], pd.DataFrame, List[dict]]:
        """
        Implements a function that, given a dataset, partitions the data for cross-validation.
        Also calls standardization, stratification, and drops excluded columns
        """
        #drop excluded columns
        if exclude_columns != []:
            data = data.drop(exclude_columns,axis=1)

        #stratify data
        if stratify and data[target_column].dtype == 'object':
            stratify_data = data[target_column]
        else:
            stratify_data = None

        # downcast to smaller float and ints
        if downcast:
            data = self.downcast_data(data)

        # Split 80% for training/testing and 20% for final validation
        train_val, test = train_test_split(data, test_size=0.2, stratify=stratify_data)
        if cv_type == '5x2':
            splits = []
            for _ in range(5):
                split = train_test_split(train_val, test_size=0.5, stratify=stratify_data)
                split = self.standardize(split[0],split[1],standardize_columns)
                splits.extend(split)
            return splits, test, self.standardize_mappings
        elif cv_type == '10-fold':
            kf = KFold(n_splits=10, shuffle=True)
            splits = list(kf.split(train_val))
            for split in splits:
                split = self.standardize(split[0],split[1],standardize_columns)
            return splits, test, self.standardize_mappings
        