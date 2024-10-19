import numpy as np
import copy
import pandas as pd
import json
from dataLogger import debug_log
from typing import Union, Tuple, List
from dataSplitifyzer import DataSplitifyzer
from dataPreprocessor import DataPreprocessor

class ModelId3Classifier:

    def __init__(self, log_enabled: bool = True):
        self.splitifyzer = DataSplitifyzer()
        self.eg_unpruned_tree = {}
        self.eg_pruned_tree = {}
        self.ordinal_mappings = {}
        self.log_enabled = log_enabled
        

    def evaluate(self, y_true: Union[np.ndarray, pd.Series], 
                 y_pred: Union[np.ndarray, pd.Series], 
                 task: str = 'classification') -> float:
        """
        EVALUATION FUNCTION
        Calculate error metric from ground truth and predictions series
        """
        if task == 'classification':
            y_true = y_true.reset_index(drop=True)
            y_pred = y_pred.reset_index(drop=True)
            return np.mean(y_true != y_pred)

    def calc_entropy(self, data: pd.DataFrame, target_column: str) -> float:
        """
        HELPER FUNCTION

        Calculate Entropy, H(S)
        H(S) = sum(-p_i * log_2 (p_i))
        p_i is the probability (proportion) of class i in this subset of the data.
        """
        proportions = data[target_column].value_counts(normalize=True)
        return float(-sum(p * np.log2(p) for p in proportions))

    def calculate_info_gain_all_attr(self, data: pd.DataFrame, 
                                     target_column: str, 
                                     exclude_columns: List[str] = []) -> Tuple[dict, dict]:
        """
        HELPER FUNCTION

        Take dataframe and calculate info gain of all attribute columns
        Information gain (per attribute)
        Gain(a) = H(S) - sum( (count(S_aj) / count(S) * H(S_aj))
        S_aj are the samples of a given value (j) of an attribute (a)

        Return calculated gains of each attribute, all values for each attribute
        """
        col_list = set(data.columns) - set(exclude_columns) - {target_column}
        entropy_S = self.calc_entropy(data, target_column)
        count_S = len(data)

        gain_dict = {}
        attr_val_dict = {}
        for attribute in col_list:
            gain_loss = 0
            attr_val_dict[attribute] = data[attribute].unique().tolist()
            grouped_df = data.groupby(attribute)
            for _, df in grouped_df:
                count_S_aj = len(df)
                H_S_aj = self.calc_entropy(df, target_column)
                gain_loss += (count_S_aj / count_S) * H_S_aj
            gain_dict[attribute] = entropy_S - gain_loss

        return gain_dict, attr_val_dict

    def add_node(self, parent_node: dict, data: pd.DataFrame, 
                 target_column: str, exclude_columns: List[str] = []) -> None:
        """
        TREE BUILDING FUNCTION
        Recursively add nodes to the decision tree
        """
        # Base case: if all samples have the same class (no decision necessary)
        if len(data[target_column].unique()) == 1:
            parent_node['leaf'] = data[target_column].iloc[0]
            return

        # Base case: if no features left to split on (no decision possible)
        if len(data.columns) == 1:
            parent_node['leaf'] = data[target_column].mode().iloc[0]
            return

        # Calculate gain of each attribute
        gain_dict, attr_val_dict = self.calculate_info_gain_all_attr(data, target_column, exclude_columns)

        if not gain_dict:  # No more attributes to split on
            parent_node['leaf'] = data[target_column].mode().iloc[0]
            return

        # Find the attribute that gives the maximum gain
        max_gain_attribute = max(gain_dict, key=gain_dict.get)
        parent_node[max_gain_attribute] = {}

        # Add a node for each value in the max gain attribute
        for attr_value in attr_val_dict[max_gain_attribute]:
            subset = data[data[max_gain_attribute] == attr_value].drop(columns=[max_gain_attribute])
            parent_node[max_gain_attribute][attr_value] = {}
            
            if subset.empty:
                parent_node[max_gain_attribute][attr_value]['leaf'] = data[target_column].mode().iloc[0]
            else:
                self.add_node(parent_node[max_gain_attribute][attr_value], subset, target_column, 
                              exclude_columns + [max_gain_attribute])

    def build_tree(self, data: pd.DataFrame, target_column: str, 
                   exclude_columns: List[str] = []) -> dict:
        """
        TREE BUILDING FUNCTION
        Build the decision tree using the ID3 algorithm
        """
        decision_tree = {}
        self.add_node(decision_tree, data, target_column, exclude_columns)
        return decision_tree

    def predict_sample(self, sample: pd.Series, tree: dict = None) -> str:
        """
        MODEL EVALUATION FUNCTION
        Predict the class for a single sample. In edge case where a value wasn't present in training data, return most common class.
        """

        if 'leaf' in tree:
            return tree['leaf']

        attribute = next(iter(tree))
        value = sample[attribute]

        if value not in tree[attribute]:
            # Return the most common class if the value is not in the tree
            return max((self.predict_sample(sample, subtree) 
                        for subtree in tree[attribute].values()), 
                       key=lambda x: sum(1 for v in tree[attribute].values() 
                                         if self.predict_sample(sample, v) == x))

        return self.predict_sample(sample, tree[attribute][value])

    def prune_tree(self, node: dict, pruning_data: pd.DataFrame, target_column: str):
        """
        PRUNING FUNCTION
        Prune the decision tree using the pruning dataset
        """
        if 'leaf' in node:
            return node

        if pruning_data.empty:
            # If no pruning data reaches this node, don't prune
            return node

        attribute = list(node.keys())[0]
        
        for attr_value, child in node[attribute].items():
            if isinstance(child, dict) and 'leaf' not in child:
                # Filter pruning data for this branch
                child_pruning_data = pruning_data[pruning_data[attribute] == attr_value]
                self.prune_tree(child, child_pruning_data, target_column)

        # Check if pruning this node improves accuracy
        unpruned_accuracy = self.evaluate_subtree(node, pruning_data, target_column)
        
        if pruning_data[target_column].empty:
            # If there's no data to determine majority class, don't prune
            return node
        
        majority_class = pruning_data[target_column].mode().iloc[0]
        pruned_accuracy = self.evaluate_leaf(majority_class, pruning_data, target_column)

        if pruned_accuracy > unpruned_accuracy:
            node.clear()
            node['leaf'] = majority_class

        return node


    def evaluate_subtree(self, node: dict, data: pd.DataFrame, target_column: str) -> int:
        """
        PRUNING FUNCTION
        Evaluate the performance of a subtree on the given data
        Returns the number of correct predictions
        """
        predictions = [self.predict_sample(row, node) for _, row in data.iterrows()]
        return sum(predictions == data[target_column])

    def evaluate_leaf(self, leaf_value: Union[str, int], data: pd.DataFrame, target_column: str) -> int:
        """
        PRUNING FUNCTION
        Evaluate the performance of a leaf node on the given data
        Returns the number of correct predictions if this node were a leaf
        """
        return sum(data[target_column] == leaf_value)

    def run_tree_predictions(self, tree:dict, test_data:pd.DataFrame) -> Union[np.ndarray, pd.Series]:
        """
        EVALUATION FUNCTION
        Run the tree
        """
        predictions = [self.predict_sample(row,tree) for _, row in test_data.iterrows()]
        return pd.Series(predictions)

    def run_id3_model(self, data: pd.DataFrame, target_column: str, task: str, 
                    standardize_columns: List[str] = [], 
                    exclude_columns: List[str] = []) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        RUN ME to run the 5x2 cross validation one time
        Runs the ID3 model on the dataframe, prunes, and evaluates models on each fold and pruning (holdout) data.

        args:
            data (pd.DataFrame): The input data containing features and the target column.
            target_column: The name of the target column in the dataset.
            task: The type of task, either 'classification' or 'regression'. Will always be 'classification' for this model.
            standardize_columns: List of columns to standardize.
            exclude_columns: List of columns to exclude from the model.
        Returns:
            4x List[float]: unpruned_scores, pruned_scores, unpruned_holdout_scores, pruned_holdout_scores.
        """

        # Split data into 5xsplit of 80% for training/cv and 20% for pruning (holdout)
        splits, pruning_data, _ = self.splitifyzer.split_data(data, target_column, cv_type='5x2', 
                                                            stratify=True, 
                                                            standardize_columns=[],
                                                            exclude_columns=exclude_columns)

        unpruned_scores = []
        pruned_scores = []
        unpruned_holdout_scores = []
        pruned_holdout_scores = []

        for i in range(0, len(splits), 2):
            debug_log(self.log_enabled, f"Running ID3 on folds {i}, {i+1}...")

            train_data, val_data = splits[i], splits[i+1]

            # Build and evaluate unpruned tree A
            unpruned_tree = self.build_tree(data=train_data,target_column=target_column,exclude_columns=[])
            unpruned_val_y_pred = self.run_tree_predictions(tree=unpruned_tree,test_data=val_data)
            unpruned_test_score = self.evaluate(val_data[target_column],unpruned_val_y_pred)
            unpruned_scores.append(float(unpruned_test_score))

            # Prune a copy of tree A
            pruned_tree = copy.deepcopy(unpruned_tree)
            pruned_tree = self.prune_tree(pruned_tree, pruning_data, target_column)
            
            # Evaluate pruned tree A
            pruned_val_y_pred = self.run_tree_predictions(tree=pruned_tree,test_data=val_data)
            pruned_test_score = self.evaluate(val_data[target_column],pruned_val_y_pred)
            pruned_scores.append(float(pruned_test_score))

            # Evaluate unpruned tree A on the validation (holdout) data
            unpruned_hold_y_pred = self.run_tree_predictions(tree=unpruned_tree,test_data=pruning_data)
            unpruned_test_score = self.evaluate(pruning_data[target_column],unpruned_hold_y_pred)
            unpruned_holdout_scores.append(float(unpruned_test_score))

            

            # Evaluate pruned tree A on the validation (holdout) data
            pruned_hold_y_pred = self.run_tree_predictions(tree=pruned_tree,test_data=pruning_data)
            pruned_test_score = self.evaluate(pruning_data[target_column],pruned_hold_y_pred)
            pruned_holdout_scores.append(float(pruned_test_score))

            # *************SWAP TRAIN AND VAL DATA, then RUN SAME METHODS**************
            train_data, val_data = val_data, train_data

            # Build and evaluate unpruned tree B
            unpruned_tree = self.build_tree(data=train_data,target_column=target_column,exclude_columns=[])
            unpruned_val_y_pred = self.run_tree_predictions(tree=unpruned_tree,test_data=val_data)
            unpruned_test_score = self.evaluate(val_data[target_column],unpruned_val_y_pred)
            unpruned_scores.append(float(unpruned_test_score))

            # Prune a copy of tree B
            pruned_tree = copy.deepcopy(unpruned_tree)
            pruned_tree = self.prune_tree(pruned_tree, pruning_data, target_column)

            # Evaluate pruned tree B
            pruned_val_y_pred = self.run_tree_predictions(tree=pruned_tree,test_data=val_data)
            pruned_test_score = self.evaluate(val_data[target_column],pruned_val_y_pred)
            pruned_scores.append(float(pruned_test_score))

            # Evaluate unpruned tree B on the validation (holdout) data
            unpruned_hold_y_pred = self.run_tree_predictions(tree=unpruned_tree,test_data=pruning_data)
            unpruned_test_score = self.evaluate(pruning_data[target_column],unpruned_hold_y_pred)
            unpruned_holdout_scores.append(float(unpruned_test_score))

            # Evaluate pruned tree B on the validation (holdout) data
            pruned_hold_y_pred = self.run_tree_predictions(tree=pruned_tree,test_data=pruning_data)
            pruned_test_score = self.evaluate(pruning_data[target_column],pruned_hold_y_pred)
            pruned_holdout_scores.append(float(pruned_test_score))

            # retain an example pruned and unpruned tree for display
            self.eg_pruned_tree = pruned_tree
            self.eg_unpruned_tree = unpruned_tree

        debug_log(self.log_enabled, f"Done running ID3.")
        return unpruned_scores, pruned_scores, unpruned_holdout_scores, pruned_holdout_scores

    def read_pre_cfg(self, file_path: str):
        """
        HELPER FUNCTION
        Read the .pre_cfg file and store the inverse ordinal mappings for printing trees
        """
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        self.ordinal_mappings = config.get('ordinal_mappings', {})
        
        # Invert mappings for easier lookup while printing the tree
        for column, mapping in self.ordinal_mappings.items():
            self.ordinal_mappings[column] = {v: k for k, v in mapping.items()}

    def print_decision_tree(self, tree:dict=None, indent:str="") -> None:
        """
        DEBUG FUNCTION
        Print the decision tree in a readable format, using text values from ordinal mappings
        """
        if tree is None:
            tree = self.eg_unpruned_tree

        for key, value in tree.items():
            if key == 'leaf':
                leaf_value = self.ordinal_mappings.get('class', {}).get(value, value)
                print(f"{indent}Predict: {leaf_value}")
            else:
                print(f"{indent}{key}:")
                for attr_value, subtree in value.items():
                    mapped_value = self.ordinal_mappings.get(key, {}).get(int(attr_value), attr_value)
                    print(f"{indent}  {mapped_value}:")
                    self.print_decision_tree(subtree, indent + "--> ") 

def run_stats():
    """
    RUN ME to run 5x2 validation 10 times, output the mean and stdev of each, and output aggregate mean and stdev
    """
    all_unpruned_scores, all_pruned_scores, all_unpruned_holdout_scores, all_pruned_holdout_scores = [],[],[],[]
    print("unpruned cross-val mean, unpruned cross-val stdev, unpruned holdout mean,  unpruned holdout stdev,", 
          "pruned cross-val mean, pruned cross-val stdev, pruned cross-val holdout mean, pruned cross-val holdout stdev")
    for i in range(0,10):
        preprocessor = DataPreprocessor(False)
        data = preprocessor.read_preprocessed_data("car_preprocessed.csv")

        model = ModelId3Classifier(log_enabled=False)
        model.read_pre_cfg("car.pre_cfg")
        
        unpruned_scores, pruned_scores, unpruned_holdout_scores, pruned_holdout_scores = model.run_id3_model(data, 'class', 'classification')
        all_unpruned_scores.extend(unpruned_scores)
        all_pruned_scores.extend(pruned_scores)
        all_unpruned_holdout_scores.extend(unpruned_holdout_scores)
        all_pruned_holdout_scores.extend(pruned_holdout_scores)
        

        print(f"{np.mean(unpruned_scores):.4f}, {np.std(unpruned_scores):.4f},",
              f"{np.mean(unpruned_holdout_scores):.4f}, {np.std(unpruned_holdout_scores):.4f},",
              f"{np.mean(pruned_scores):.4f}, {np.std(pruned_scores):.4f},",
              f"{np.mean(pruned_holdout_scores):.4f}, {np.std(pruned_holdout_scores):.4f}")
    print()

    print(f"unpruned cross-validation error rate: mean= {np.mean(all_unpruned_scores):.4f}, stdev= {np.std(all_unpruned_scores):.4f} n(scores)= {len(all_unpruned_scores)}")
    print(f"pruned cross-validation error rate: {np.mean(all_pruned_scores):.4f}, stdev= {np.std(all_pruned_scores):.4f}, n(scores) = {len(all_pruned_scores)}")
    print(f"unpruned holdout error rate: {np.mean(all_unpruned_holdout_scores):.4f}, stdev= {np.std(all_unpruned_holdout_scores):.4f}, n(scores) = {len(all_unpruned_holdout_scores)}")
    print(f"pruned holdout error rate: {np.mean(all_pruned_holdout_scores):.4f}, stdev= {np.std(all_pruned_holdout_scores):.4f}, n(scores) = {len(all_pruned_holdout_scores)}")


def run_once():
    """
    RUN ME to run the ID3 model with 5x2 cross-validation once, and output a sample decision tree
    """

    preprocessor = DataPreprocessor(True)
    data = preprocessor.read_preprocessed_data("car_preprocessed.csv")

    model = ModelId3Classifier()
    model.read_pre_cfg("car.pre_cfg")
    
    unpruned_scores, pruned_scores, unpruned_holdout_scores, pruned_holdout_scores = model.run_id3_model(data, 'class', 'classification')
 
    print(f"Average unpruned cross-validation error rate: {np.mean(unpruned_scores):.4f}, n(scores) = {len(unpruned_scores)}")
    print(f"Average pruned cross-validation error rate: {np.mean(pruned_scores):.4f}, n(scores) = {len(pruned_scores)}")

    print(f"Average unpruned holdout error rate: {np.mean(unpruned_holdout_scores):.4f}, n(scores) = {len(unpruned_holdout_scores)}")
    print(f"Average pruned holdout error rate: {np.mean(pruned_holdout_scores):.4f}, n(scores) = {len(pruned_holdout_scores)}")

    print("\nPruned Decision Tree Structure:")
    model.print_decision_tree(model.eg_pruned_tree)

if __name__ == "__main__":
    run_once()
    # main()
    run_stats()