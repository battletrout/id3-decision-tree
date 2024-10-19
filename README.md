# ID3 Decision Tree Classifier for UCI Car Evaluation Dataset

This project implements the Iterative Dichotomiser 3 (ID3) decision tree algorithm to create a classifier on the Car Evaluation Database dataset (N=1728). The decision tree is built using Shannon Information Gain as the splitting criteria, and then pruned using reduced-error pruning. The performance of pruned and unpruned decision trees are compared using 5x2 cross-validation, repeated 10 times. 

This is a relatively simple dataset, so over-fitting didn't seem to be an issue. Unpruned trees performed slightly better than pruned trees on my run: Mean Classification Error of  0.099 (SD=0.013) for unpruned trees and 0.104 (SD=0.014) for pruned trees. 

The sample dataset is the car evaluation dataset from UC Irvine's Machine Learning Repository, "originally developed for the demonstration of DEX, M. Bohanec, V. Rajkovic: Expert system for decision making. Sistemica 1(1), pp. 145-157, 1990.)":
https://archive.ics.uci.edu/dataset/19/car+evaluation

## Files Description

- `data/`: Directory containing the dataset and related files
  - `car.data`: Raw dataset
  - `car.names`: Description of the dataset
  - `car.pre_cfg`: Configuration file for preprocessing
  - `car_preprocessed.csv`: Preprocessed dataset
- `dataLogger.py`: Logging utilities for the project
- `dataPreprocessor.py`: Data preprocessing module
- `dataSplitifyzer.py`: Module for splitting and stratifying data
- `model_id3_decision_tree.py`: Main ID3 decision tree model implementation
- `requirements.txt`: List of Python dependencies

## Setup and Installation

1. Clone the repository
2. Install the required packages:
```
pip install -r requirements.txt
```
## Usage

To run the ID3 model on the car evaluation dataset:

```
python model_id3_decision_tree.py
```
This will perform 5x2 cross-validation, build both pruned and unpruned decision trees, and output the results.

```
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
```
Note
This project uses Windows-style file paths. If you're running on a Unix-based system, you may need to modify the file paths in the Python scripts.
