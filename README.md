# Titanic Dataset — Data Cleaning & Preprocessing Pipeline

This repository contains Python scripts for cleaning and preprocessing the Titanic dataset to prepare it for machine learning tasks. The project covers key steps such as handling missing values, encoding categorical variables, outlier detection/removal, feature scaling, and train-test splitting.

---

## Project Overview

Data preprocessing is a crucial step before applying any machine learning algorithm. This project demonstrates best practices to transform raw Titanic data into a clean, structured format ready for model training.

---

## Files and Workflow

| File          | Description                                                  |
|---------------|--------------------------------------------------------------|
| `clean.py`    | Initial data cleaning: handling missing values, feature engineering, data type conversion, and saving cleaned data. |
| `process.py`  | Further preprocessing: outlier detection/removal, encoding, scaling, train-test split, and saving final datasets. |
| `cleaned_data.csv`       | Output from `clean.py` —  | Final cleaned and preprocessed dataset ready for ML.           |
| `X_train.csv`, `y_train.csv` | Training features and labels after preprocessing and splitting. |

---

## Step-by-Step Summary

### Step 1: Initial Cleaning (`clean.py`)
- Load raw Titanic dataset.
- Inspect dataset for missing values and duplicates.
- Impute missing values:
  - Fill missing `Age` with median.
  - Fill missing `Embarked` with mode.
- Create new binary feature `HasCabin` indicating presence of cabin data.
- Drop original `Cabin` column.
- Convert relevant columns (`Pclass`, `Sex`, `Embarked`, `Survived`) to categorical types.
- Save cleaned data to `cleaned_data.csv`.

### Step 2: Preprocessing (`process.py`)
- Load cleaned data from `cleaned_data.csv`.
- Drop irrelevant columns: `PassengerId`, `Name`, `Ticket`.
- Visualize and detect outliers in `Age` and `Fare` using boxplots.
- Remove outliers using the Interquartile Range (IQR) method.
- Encode categorical variables:
  - Label encode `Sex`.
  - One-hot encode `Embarked` and `Pclass`.
- Standardize numerical features (`Age`, `Fare`) using `StandardScaler`.
- Split dataset into training and test sets (80% train, 20% test).
- Save processed datasets as `X_train.csv`, `y_train.csv`, and `cleaned_data_final.csv`.

---

## Dependencies

Make sure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
