# Cedit-Card-Fraud-Detection
This project aims to develop a machine learning model capable of identifying fraudulent transactions from a credit card transaction dataset. The main challenge lies in the severe class imbalance, with only 0.17% of transactions being fraudulent.

Objectives:

Handle class imbalance using SMOTE and Undersampling techniques.
Compare the performance of Logistic Regression, Random Forest, and XGBoost.
Evaluate the models using appropriate metrics (Recall, Precision, F1-score, ROC and PR curves).
2. Dataset
The dataset used is "Credit Card Fraud Detection" available on Kaggle.

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Features: 284,807 transactions, 30 variables (V1-V28 from PCA, Time, Amount).
Target: Class (0 = Normal, 1 = Fraud).
Note: To run the notebook, please download the creditcard.csv file from the Kaggle link above and place it in the same directory as the notebook.

3. Methodology
The project was conducted in several stages:

Exploratory Data Analysis: Visualizing distributions and analyzing correlations.
Preprocessing: Normalizing the Time and Amount variables, performing a stratified train-test split.
Handling Imbalance: Applying Random Undersampling and SMOTE to the training set.
Modeling: Training and evaluating three different algorithms on the various data versions.
Final Evaluation: Visually comparing the best models using ROC and PR curves.
4. Results
The model comparison showed that the SMOTE technique was the most effective for handling the imbalance. The XGBoost model provided the best overall trade-off.

Model & Strategy	Recall	Precision	F1-Score
Log. Regression + SMOTE	0.92	0.06	0.11
Random Forest + SMOTE	0.83	0.85	0.84
XGBoost + SMOTE	0.88	0.74	0.80
(Insert your most important charts here, like the PR curve)

5. How to Run the Project
Clone this repository.
Install the dependencies: pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
Download the dataset from Kaggle and place creditcard.csv in the project's root directory.
Launch Jupyter Notebook and open the .ipynb file.
