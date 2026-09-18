# Week 6 – Integrative Capstone Project and Evaluation

## Project
**Titanic Survival Prediction using Machine Learning**

## Objective
Build a complete Python data-science pipeline covering data acquisition, cleaning, EDA, feature engineering, supervised modeling, and evaluation.

## Dataset
Titanic `train.csv` from the public Kaggle Titanic: Machine Learning from Disaster dataset.

## Models
- Logistic Regression
- Random Forest

## Features
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone.

## Preprocessing
- Remove duplicate rows
- Median imputation for numerical values
- Most-frequent imputation for categorical values
- One-hot encoding
- Standard scaling
- Scikit-learn Pipeline/ColumnTransformer to reduce data leakage

## Evaluation
Accuracy, precision, recall, F1-score, ROC-AUC, classification report, and confusion matrix.

## How to Run
1. Place `train.csv` in this folder.
2. Install:
   `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run:
   `python week6_capstone_titanic.py`

The script prints the exact test metrics for both models and displays confusion matrices.
