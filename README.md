# ID3 Decision Tree Classifier for Car Evaluation

This experiment implements the Iterative Dichotomiser 3 (ID3) decision tree algorithm to create a classifier on the Car Evaluation Database dataset (N=1728). The decision tree is built using Shannon Information Gain as the splitting criteria, and then pruned using reduced-error pruning. The performance of pruned and unpruned decision trees are compared using 5x2 cross-validation, repeated 10 times. The unpruned models slightly outperform the pruned models on cross-validation data, with Mean Classification Errors of 0.099 (SD=0.013) and 0.104 (SD=0.014) respectively. Pruned models show significantly better performance on the pruning dataset (mean error=0.062, SD=0.013) compared to unpruned models (mean error=0.100, SD=0.018). The results suggest that for this relatively small dataset with few features, pruning may not significantly improve generalization, but it does optimize performance on the data used to prune the tree.

The sample dataset is the car evaluation dataset from UC Irvine's Machine Learning Repository
https://archive.ics.uci.edu/dataset/19/car+evaluation

## Files Description

- `data/`: Directory containing the dataset and related files
  - `car.data`: Raw dataset
  - `car.names`: Description of the dataset
  - `car.pre_cfg`: Configuration file for preprocessing
  - `car_preprocessed.csv`: Preprocessed dataset
- `.gitignore`: Specifies intentionally untracked files to ignore
- `dataLogger.py`: Logging utilities for the project
- `dataPreprocessor.py`: Data preprocessing module
- `dataSplitifyzer.py`: Module for splitting and stratifying data
- `model_id3_decision_tree.py`: Main ID3 decision tree model implementation
- `requirements.txt`: List of Python dependencies

## Setup and Installation

1. Clone the repository
2. Install the required packages:
pip install -r requirements.txt

## Usage

To run the ID3 model on the car evaluation dataset:

```python
python model_id3_decision_tree.py
This will perform 5x2 cross-validation, build both pruned and unpruned decision trees, and output the results.
Data Preprocessing
The DataPreprocessor class in dataPreprocessor.py handles initial data preprocessing, including:

  Encoding ordinal data
  Handling missing values (if any)
  Converting the data to a suitable format (.csv with headers) for the model

Data Splitting
The DataSplitifyzer class in dataSplitifyzer.py is responsible for:

  Splitting the data for cross-validation
  Stratifying the data
  Optionally standardizing features

Model Implementation
The ModelId3Classifier class in model_id3_decision_tree.py implements the ID3 algorithm, including:

  Building the decision tree
  Pruning the tree
  Making predictions
  Evaluating model performance

Note
This project uses Windows-style file paths. If you're running on a Unix-based system, you may need to modify the file paths in the Python scripts.

