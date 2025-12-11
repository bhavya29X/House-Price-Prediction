# House-Price-Prediction
End-to-end machine learning project for predicting housing prices using Python and Scikit-learn. Includes data preprocessing pipelines, model training, evaluation (RMSE), and comparison between Linear Regression, Decision Tree, and Random Forest models. 

### 🏡 Housing Price Prediction — End-to-End ML Project

##### This project predicts housing prices based on multiple features using Python and Scikit-learn. It includes a full machine learning workflow with preprocessing, pipelines, model comparison, and evaluation.

### 🚀 Tech Stack

Language: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

### 📌 Project Workflow

1. Loaded and explored the housing dataset (EDA).
2. Applied stratified train–test split using an income category to maintain distribution.
3. Separated features and labels (target = median house value).
4. Identified numerical and categorical columns.
5. Built preprocessing pipelines using:
   - Numerical: Median imputation + Standard scaling
   - Categorical: One-Hot Encoding
6. Applied the pipelines to create clean, model-ready data.
7. Trained and compared three regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
8. Evaluated models using RMSE, MAE, and R² (if included).
9. Selected Random Forest as the best-performing model.

### 📈 Model Performance

Random Forest delivered the lowest RMSE and showed strong predictive performance on unseen data.

### 🎯 Key Learnings

 - Building reusable ML pipelines with Scikit-learn

 - Importance of stratified sampling for fair evaluation

 - Understanding model behavior across different algorithms

 - Using RMSE to objectively compare regression models
