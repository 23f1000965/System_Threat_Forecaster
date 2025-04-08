# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # data visualization
from sklearn.model_selection import train_test_split, GridSearchCV  # data splitting
from sklearn.preprocessing import StandardScaler  # data preprocessing
from sklearn.feature_selection import SelectKBest, f_classif  # feature selection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # model evaluation
import xgboost as xgb  # model building
from sklearn.svm import SVC  # model building
import lightgbm as lgb  # model building
import pickle  # model saving
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load the training and test datasets
df = pd.read_csv('/kaggle/input/System-Threat-Forecaster/train.csv')
X_test = pd.read_csv('/kaggle/input/System-Threat-Forecaster/test.csv')

# Check the shape of the training data
print(df.shape)

# Check the shape of the test data
print(X_test.shape)

# Display the first few rows of the training data
print(df.head())

# Display the first few rows of the test data
print(X_test.head())

# Get information about the training data
df.info()

# Get information about the test data
X_test.info()

# Summary statistics
print(df.describe(include='all'))

# Check for missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])

# Plot missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values[missing_values > 0].index, y=missing_values[missing_values > 0])
plt.title('Missing Values in Training Data')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()

# Distribution of the target variable
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Handling Missing Values
for col in df.columns:
    if col == 'target':
        df.loc[:, col] = df[col].fillna(df[col].median())
        continue
    if df[col].dtype in ['float64', 'int64']:
        df.loc[:, col] = df[col].fillna(df[col].median())
        X_test.loc[:, col] = X_test[col].fillna(df[col].median())
    else:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])
        X_test.loc[:, col] = X_test[col].fillna(df[col].mode()[0])

# Encode Categorical Features
df_encoded = pd.get_dummies(df, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align train and test datasets to ensure they have the same feature set
df_encoded, X_test_encoded = df_encoded.align(X_test_encoded, join='inner', axis=1)

# Feature Selection
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
selector = SelectKBest(score_func=f_classif, k=40)
X_selected = selector.fit_transform(df_encoded.drop(columns=['target'], errors='ignore'), df['target'])
X_test_selected = selector.transform(X_test_encoded)

# Standardize Numeric Features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)
X_test_selected = scaler.transform(X_test_selected)

# Extract Target Variable and Features
y = df['target']

# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Display the shapes of the training and validation sets
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Build and Train LightGBM Model
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Hyperparameter Tuning
param_grid = {
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}
random_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
random_search.fit(X_train, y_train)

# Best Model
best_model = random_search.best_estimator_

# Evaluate the Best Model
y_val_pred_best = best_model.predict(X_val)
best_accuracy = accuracy_score(y_val, y_val_pred_best)
print(f"Best Validation Accuracy: {best_accuracy:.2f}")
print("\nBest Classification Report:")
print(classification_report(y_val, y_val_pred_best))

# Confusion Matrix for Best Model
cm_best = confusion_matrix(y_val, y_val_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Make Predictions on Test Data
y_pred = best_model.predict(X_test_selected)