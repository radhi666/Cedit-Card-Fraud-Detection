# 💳 Credit Card Fraud Detection using Machine Learning

## 1. Project Overview

This project aims to develop a machine learning model capable of identifying fraudulent transactions from a credit card transaction dataset. The main challenge lies in the severe class imbalance, with only **0.17%** of transactions being fraudulent.

### Objectives

- Handle class imbalance using **SMOTE** and **Random Undersampling** techniques.
- Compare the performance of **Logistic Regression**, **Random Forest**, and **XGBoost**.
- Evaluate the models using appropriate metrics:
  - Recall
  - Precision
  - F1-Score
  - ROC Curve
  - Precision-Recall Curve

---

## 2. Dataset

The dataset used is **Credit Card Fraud Detection**, available on Kaggle.

- **Dataset Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Features:** 284,807 transactions with 30 variables:
  - V1–V28 (PCA-transformed features)
  - Time
  - Amount
- **Target Variable:** `Class`
  - 0 = Legitimate Transaction
  - 1 = Fraudulent Transaction

> **Note:** Download the `creditcard.csv` file from Kaggle and place it in the project's root directory before running the notebook.

---

## 3. Methodology

The project was conducted through the following stages:

### 3.1 Exploratory Data Analysis (EDA)

- Analyze class distribution.
- Visualize transaction patterns.
- Study feature correlations.

### 3.2 Data Preprocessing

- Normalize the **Time** and **Amount** features.
- Perform a **Stratified Train-Test Split**.

### 3.3 Handling Class Imbalance

Two techniques were explored:

- Random Undersampling
- SMOTE (Synthetic Minority Oversampling Technique)

### 3.4 Model Training

Three machine learning algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### 3.5 Performance Evaluation

Models were compared using:

- Recall
- Precision
- F1-Score
- ROC Curve
- Precision-Recall Curve

---

## 4. Results

The experiments demonstrated that **SMOTE** significantly improved model performance on the imbalanced dataset.

Among all tested approaches, **XGBoost + SMOTE** provided the best overall balance between fraud detection capability and prediction precision.

| Model & Strategy | Recall | Precision | F1-Score |
|-----------------|---------|-----------|----------|
| Logistic Regression + SMOTE | 0.92 | 0.06 | 0.11 |
| Random Forest + SMOTE | 0.83 | 0.85 | 0.84 |
| XGBoost + SMOTE | 0.88 | 0.74 | 0.80 |

### Key Insights

- Logistic Regression achieved the highest recall but generated many false positives.
- Random Forest delivered the highest F1-score.
- XGBoost offered a strong trade-off between recall and precision, making it the most balanced model.

### Performance Curves

Insert your best visualizations here:

- ROC Curve
- Precision-Recall Curve
- Confusion Matrix

Example:

```markdown
![PR Curve](images/pr_curve.png)

![ROC Curve](images/roc_curve.png)
