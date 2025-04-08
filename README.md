# System Threat Forecaster

This repository contains a Pyhton Notebook for building a machine learning pipeline to predict system threats. The notebook includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Notebook Workflow](#notebook-workflow)
  - [1. Data Loading](#1-data-loading)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Feature Engineering](#4-feature-engineering)
  - [5. Model Training and Evaluation](#5-model-training-and-evaluation)
  - [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
  - [7. Submission File Creation](#7-submission-file-creation)
- [Models Used](#models-used)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Overview

The goal of this project is to predict system threats using machine learning models. The notebook demonstrates the end-to-end process of building a predictive model, from data loading to generating predictions for submission.

---

## Dataset

The dataset consists of training and test CSV files:
- **Training Data**: Contains labeled data for model training.
- **Test Data**: Contains unlabeled data for prediction.

The data is loaded from the Kaggle input directory:
- Training file: `/kaggle/input/System-Threat-Forecaster/train.csv`
- Test file: `/kaggle/input/System-Threat-Forecaster/test.csv`

---

## Environment Setup

This notebook is designed to run in a Kaggle environment with pre-installed libraries. The following Python libraries are used:
- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`: Data visualization
- `scikit-learn`: Machine learning utilities
- `xgboost`, `lightgbm`: Model building
- `pickle`: Model saving

---

## Notebook Workflow

### 1. Data Loading
The training and test datasets are loaded using `pandas.read_csv`. The shapes of the datasets are checked, and the first few rows are displayed to understand the structure of the data.

### 2. Exploratory Data Analysis (EDA)
EDA is performed to understand the data better:
- **Summary Statistics**: Using `df.describe(include='all')`.
- **Missing Values**: Checked using `df.isnull().sum()` and visualized with a bar plot.
- **Target Variable Distribution**: Visualized using a count plot.
- **Correlation Analysis**: A heatmap is plotted to analyze relationships between features.

### 3. Data Preprocessing
- **Handling Missing Values**: Numeric columns are filled with the median, and categorical columns are filled with the mode.
- **Encoding Categorical Features**: One-hot encoding is applied using `pd.get_dummies`.
- **Feature Alignment**: Ensures the training and test datasets have the same feature set.

### 4. Feature Engineering
- **Feature Selection**: The top 40 features are selected using `SelectKBest` with the `f_classif` scoring function.
- **Feature Scaling**: Numeric features are standardized using `StandardScaler`.

### 5. Model Training and Evaluation
The dataset is split into training and validation sets using `train_test_split`. Several models are trained and evaluated:
- **LightGBM**
- **XGBoost**
- **Support Vector Classifier (SVC)**
- **Random Forest**

Each model is evaluated using:
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix**

### 6. Hyperparameter Tuning
Hyperparameter tuning is performed using `GridSearchCV` or `RandomizedSearchCV` to find the best model configuration.

### 7. Submission File Creation
Predictions on the test dataset are saved to a CSV file for submission.

---

## Models Used

The following models are implemented:
1. **LightGBM**: A gradient boosting framework.
2. **XGBoost**: An optimized gradient boosting library.
3. **Support Vector Classifier (SVC)**: A classification algorithm based on support vector machines.
4. **Random Forest**: An ensemble learning method using decision trees.

---

## Results

The best model is selected based on validation accuracy. The confusion matrix and classification report provide insights into the model's performance.

---

## Future Improvements

- Perform feature engineering to create new features.
- Experiment with additional models like CatBoost or neural networks.
- Use cross-validation for more robust evaluation.
- Optimize the pipeline for faster execution.

---

## How to Run

1. Clone this repository.
2. Open the notebook in a Python environment or Kaggle.
3. Run all cells sequentially to reproduce the results.

---

