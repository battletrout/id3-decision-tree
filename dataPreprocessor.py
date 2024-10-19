'''
created by NMK (Battletrout) on 20 Sep 2024
JHU EP EN.605.649

Implements most requirements of Project 1 Part 1, the remaining requirements are in
DataSplitifyzer and ModelNull.
'''

import numpy as np
import pandas as pd
import json
import os
from dataLogger import debug_log, log_function_call
from typing import Union, Tuple, List

class DataPreprocessor:    
    """
    A class for preprocessing datasets, particularly for machine learning tasks.
    
    This class provides functionality to load, clean, transform, and save data.
    It includes methods for handling missing values, encoding categorical variables,
    discretizing continuous features, and managing configuration settings.
    """

    def __init__(self, log_enabled: bool=False):
        """
        Initialize the DataPreprocessor.

        Args:
            log_enabled (bool): Whether to enable logging for this instance.

        Returns:
            None
        """
        self.log_enabled = log_enabled
        self.filepath = ""
        self.task = ""
        self.null_question_marks = True
        self.column_names = []
        self.predictor_column = ""
        self.nominal_columns = []
        self.ordinal_columns = []
        self.ordinal_mappings = {}
        self.discretize_columns = []
        self.discretize_bins = 0
        self.discretize_scheme = ""
        self.treat_as_int_columns = []

    def load_data(self, file_path: str, null_question_marks: bool = True) -> pd.DataFrame:
        """
        Load data from a CSV file and optionally replace question marks with NaN.

        Args:
            file_path (str): Path to the CSV file.
            null_question_marks (bool): If True, replace '?' with NaN.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        debug_log(self.log_enabled, f"loading file {file_path}...")
        self.filepath = file_path
         
        if null_question_marks:
            na_val = ["?"]
            self.null_question_marks = True
        else:
            na_val = []
            self.null_question_marks = False
        data = pd.read_csv(file_path, na_values=na_val)#, header=None)
        
        if self.column_names:
            data.columns = self.column_names
        debug_log(self.log_enabled, f"Done.")
        return data

    def impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataset. Replace categorical, discrete, or continuous columns
        with the most common value (mode) of that column or mean if continuous.

        Args:
            data (pd.DataFrame): Input DataFrame with missing values.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """

        debug_log(self.log_enabled, f"imputing missing values...")
        imputed_data = data.copy()
        for column in imputed_data.columns:
            missing_values = imputed_data[column].isna().sum()
            if missing_values > 0:
                if imputed_data[column].dtype == 'object' or imputed_data[column].dtype.name == 'category':
                    # For object (string) or categorical dtype, use mode
                    mode_value = imputed_data[column].mode().iloc[0]
                    imputed_data[column] = imputed_data[column].fillna(mode_value)
                    debug_log(self.log_enabled,f"Column {column} (cat): {missing_values} filled as most common category = {mode_value}")
                elif pd.api.types.is_integer_dtype(imputed_data[column]) or (column in self.treat_as_int_columns):
                    # For integer dtype, use mode
                    mode_value = imputed_data[column].mode().iloc[0]
                    imputed_data[column] = imputed_data[column].fillna(mode_value)
                    imputed_data[column] = imputed_data[column].astype(int)
                    debug_log(self.log_enabled,f"Column {column} (int): {missing_values} filled as mode = {mode_value}")
                else:
                    # For other types (assumed to be continuous), use mean as before
                    mean_value = imputed_data[column].mean()
                    imputed_data[column] = imputed_data[column].fillna(mean_value)
                    debug_log(self.log_enabled,f"Column {column} (cont): {missing_values} filled as mean = {mean_value}")
        debug_log(self.log_enabled, f"Done.")
        return imputed_data

    def encode_ordinal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode ordinal data as integers based on stored mappings.

        Args:
            data (pd.DataFrame): Input DataFrame with ordinal columns.

        Returns:
            pd.DataFrame: DataFrame with encoded ordinal columns.
        """
        debug_log(self.log_enabled, f"Encoding ordinal data on {len(self.ordinal_columns)} column(s)...")
        encoded_data = data.copy()
        for column in self.ordinal_columns:
            if column in self.ordinal_mappings:
                encoded_data[column] = encoded_data[column].map(self.ordinal_mappings[column])
        debug_log(self.log_enabled, "Done.")
        return encoded_data

    def one_hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform one-hot encoding on nominal features.

        Args:
            data (pd.DataFrame): Input DataFrame with nominal columns.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded nominal columns.
        """
        debug_log(self.log_enabled, f"One-hot-encoding on {len(self.nominal_columns)} nominal columns")
        encoded_data = data.copy()
        
        for column in self.nominal_columns:
            # Get one-hot encoding
            one_hot = pd.get_dummies(encoded_data[column], prefix=column, dtype=int)
            
            # Drop the original column and join the one-hot encoded columns
            encoded_data = encoded_data.drop(column, axis=1)
            encoded_data = encoded_data.join(one_hot)
        debug_log(self.log_enabled, "Done.")
        return encoded_data

    def discretize_features(self, data: pd.DataFrame, columns: List[str], 
                            num_bins: int, strategy: str = 'equal_width') -> pd.DataFrame:
        """
        Discretize real-valued features into bins.

        Args:
            data (pd.DataFrame): Input DataFrame.
            columns (List[str]): List of columns to discretize.
            num_bins (int): Number of bins to use for discretization.
            strategy (str): Discretization strategy ('equal_width' or 'equal_frequency').

        Returns:
            pd.DataFrame: DataFrame with discretized features.
        """
        debug_log(self.log_enabled,f"discretizing {len(columns)} columns, {num_bins} bins, {strategy}...")
        discretized_data = data.copy()
        self.discretize_bins = num_bins
        self.discretize_scheme = strategy
        for col in columns:
            if strategy == 'equal_width':
                discretized_data[col] = pd.cut(data[col], bins=num_bins, labels=False)
            elif strategy == 'equal_frequency':
                discretized_data[col] = pd.qcut(data[col], q=num_bins, labels=False)
        debug_log(self.log_enabled,"Done.")
        return discretized_data

# ******************************************************************
# Functions for saving and loading cfg and processed data from disk 
# ******************************************************************

    def save_current_cfg_file(self):
        """
        Saves the current config values of the DataPreprocessor (wrapper for save_cfg_file)
        """
        self.save_cfg_file(self.filepath, self.task, self.null_question_marks, 
                           self.column_names, self.predictor_column,
                           self.nominal_columns, self.ordinal_columns,
                           self.ordinal_mappings, self.discretize_columns,
                           self.discretize_bins, self.discretize_scheme,
                           self.treat_as_int_columns)

    def save_cfg_file(self, dataset_filename: str, task: str, null_question_marks: bool,
                      column_names: List[str], predictor_column: str, nominal_columns: List[str], 
                      ordinal_columns: List[str], ordinal_mappings: dict[str, dict[str, int]], 
                      discretize_columns:List[str], discretize_bins:int, discretize_scheme:str, 
                      treat_as_int_columns: List[str]) -> None:
        """
        Save configuration options to a JSON file.

        Args:
            dataset_filename (str): Name of the dataset file.
            task (str): Type of task (e.g., 'regression' or 'classification').
            null_question_marks (bool): Whether to replace '?' with NaN.
            column_names (List[str]): List of column names.
            predictor_column (str): Name of the predictor column.
            nominal_columns (List[str]): List of nominal column names.
            ordinal_columns (List[str]): List of ordinal column names.
            ordinal_mappings (dict[str, dict[str, int]]): Mappings for ordinal columns.
            discretize_columns (List[str]): Columns to discretize.
            discretize_bins (int): Number of bins for discretization.
            discretize_scheme (str): Discretization scheme.
            treat_as_int_columns (List[str]): Columns to treat as integers.

        Returns:
            None
        """
        config = {
            "dataset_filename": dataset_filename,
            "null_question_marks": null_question_marks,
            "task": task,
            "column_names": column_names,
            "predictor_column": predictor_column,
            "nominal_columns": nominal_columns,
            "ordinal_columns": ordinal_columns,
            "ordinal_mappings": ordinal_mappings,
            "discretize_columns": discretize_columns,
            "discretize_bins": discretize_bins,
            "discretize_scheme": discretize_scheme,
            "treat_as_int_columns": treat_as_int_columns
        }
        
        cfg_filename = f"{os.path.splitext(dataset_filename)[0]}.pre_cfg"
        
        with open(cfg_filename, 'w') as f:
            json.dump(config, f, indent=4)
        
        debug_log(self.log_enabled,f"Configuration saved to {cfg_filename}")

    def decode_ordinal_mapping(self,ordinal_map:dict) -> dict:
        """
        Decode ordinal mappings, converting string keys to integers where applicable.

        Args:
            ordinal_map (dict): Dictionary of ordinal mappings.

        Returns:
            dict: Decoded ordinal mappings.
        """
        output_dict = {}
        for p_key,p_val in ordinal_map.items():
            child_dict = {}
            for c_key,c_val in p_val.items():
                if c_key.isnumeric():
                    child_dict[int(c_key)] = c_val
                child_dict[c_key] = c_val
            output_dict[p_key] = child_dict
        return output_dict

    def read_cfg_file(self, cfg_filename: str) -> dict:
        """
        Read configuration options from a JSON file.

        Args:
            cfg_filename (str): Name of the configuration file.

        Returns:
            dict: Dictionary containing configuration options.
        """
        with open(cfg_filename, 'r') as f:
            config = json.load(f)
        
        self.filepath = config.get("dataset_filename", "")
        self.null_question_marks = config.get("null_question_marks", True)
        self.task = config.get("task","")
        self.column_names = config.get("column_names", [])
        self.predictor_column = config.get("predictor_column","")
        self.nominal_columns = config.get("nominal_columns", [])
        self.ordinal_columns = config.get("ordinal_columns", [])
        self.ordinal_mappings = config.get("ordinal_mappings", {})
        self.discretize_columns = config.get("discretize_columns", {})
        self.discretize_bins = config.get("discretize_bins",0)
        self.discretize_scheme = config.get("discretize_scheme","")
        self.treat_as_int_columns = config.get("treat_as_int_columns",[])

        if self.ordinal_mappings != {}:
            self.ordinal_mappings = self.decode_ordinal_mapping(self.ordinal_mappings)
        
        debug_log(self.log_enabled,f"Configuration loaded from {cfg_filename}")
        return config
    
    def save_preprocessed_data(self, data: pd.DataFrame, dataset_filename: str) -> None:
        """
        Save preprocessed data to a CSV file.

        Args:
            data (pd.DataFrame): Preprocessed data to save.
            dataset_filename (str): Original name of the dataset file.

        Returns:
            None
        """
        preprocessed_filename = f"{os.path.splitext(dataset_filename)[0]}_preprocessed.csv"
        
        data.to_csv(preprocessed_filename, index=False)
        
        debug_log(self.log_enabled,f"Pre-processed data saved to {preprocessed_filename}")

    def read_preprocessed_data(self, preprocessed_filename: str) -> pd.DataFrame:
        """
        Read preprocessed data from a CSV file.

        Args:
            preprocessed_filename (str): Name of the preprocessed data file.

        Returns:
            pd.DataFrame: DataFrame containing the preprocessed data.
        """
        data = pd.read_csv(preprocessed_filename)
        
        debug_log(self.log_enabled,f"Pre-processed data loaded from {preprocessed_filename}")
        return data
    
# ******************************************************************
# ********** Test Case for Cars dataset ************
# ******************************************************************

def test_car_data():
    filepath = "car.data"
    preprocessor = DataPreprocessor(True)
    preprocessor.read_cfg_file("car.pre_cfg")
    data = preprocessor.load_data(filepath,True)

    data = preprocessor.encode_ordinal(data)

    # data = preprocessor.impute_missing_values(data)
    # data = preprocessor.one_hot_encode(data)
    # data = preprocessor.discretize_features(data,preprocessor.discretize_columns,50)

    preprocessor.save_preprocessed_data(data,filepath)
    print(data.head())

    print(data['lug_boot'].dtype)
    
if __name__ == "__main__":
    test_car_data()