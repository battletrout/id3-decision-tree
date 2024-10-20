import pandas as pd
from dataLogger import debug_log
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class DataSplitifyzer:
    """
    A class for splitting, stratifying, and standardizing data for machine learning tasks.
    
    This class provides functionality to prepare data for cross-validation, including
    methods for downcasting data types, standardizing features, and splitting data
    into training and testing sets. It's designed to work with preprocessed data
    in CSV format that can be read by pandas.
    """
    def __init__(self, log_enabled:bool = False):
        """
        Initialize the DataSplitifyzer.

        Args:
            log_enabled (bool): Whether to enable logging for this instance.

        Returns:
            None
        """
        self.standardize_mappings = []
        self.log_enabled = log_enabled

    def downcast_data(self,data: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast int and float values to smaller data types where possible.
        e.g. float64 -> float32, int64 -> int16
        Args:
            data (pd.DataFrame): Input DataFrame to downcast.

        Returns:
            pd.DataFrame: DataFrame with downcasted data types.
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
        Perform z-score standardization on specified columns of training and test data.

        Args:
            train_data (pd.DataFrame): Training data to standardize.
            test_data (pd.DataFrame): Test data to standardize.
            standardize_columns (List[str]): List of column names to standardize.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Standardized training and test data.
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
        """
        Standardize validation data using pre-computed standardization mappings.

        Args:
            validation_data (pd.DataFrame): Validation data to standardize.
            standardize_columns (List[str]): List of column names to standardize.
            standardize_mappings (dict): Dictionary containing standardization parameters.

        Returns:
            pd.DataFrame: Standardized validation data.
        """    
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
        Split the data for cross-validation, applying standardization, stratification, and column exclusion as specified.

        Args:
            data (pd.DataFrame): Input data to split.
            target_column (str): Name of the target column.
            cv_type (str): Type of cross-validation ('5x2' or '10-fold').
            stratify (bool): Whether to stratify the data.
            standardize_columns (List[str]): Columns to standardize.
            exclude_columns (List[str]): Columns to exclude from the data.
            downcast (bool): Whether to downcast data types.

        Returns:
            Tuple[List[pd.DataFrame], pd.DataFrame, List[dict]]: 
                - List of DataFrames for cross-validation splits
                - Test data DataFrame
                - List of standardization mappings
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
        