# diabetes_prediction_app_ml.py

"""
Diabetes Prediction App using Traditional Machine Learning Models

This script trains and evaluates Random Forest and Linear Regression models
to predict diabetes-related outcomes using a provided dataset. The script includes
data preprocessing, model training, evaluation, and comparison.

Author: Bernardo Estacio Abrei, Fabrice Bellin, Filip Dabrowsky
Date: 15/11/2024
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Load and preprocess data
def load_and_preprocess_data(filepath):
    """
    Load the dataset and preprocess it by scaling features.

    Parameters:
        filepath (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: Scaled features (X) and target variable (y).
    """
    # Load dataset
    data = pd.read_csv(filepath)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")  # Drop unnecessary column

    # Separate features and target variable
    X = data.drop(columns=["target"])
    y = data["target"]

    # Scale features to the range [0, 1] for better model performance
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Split data into train, validation, and test sets
def split_data(X, y):
    """
    Split the data into training, validation, and testing sets.

    Parameters:
        X (ndarray): Scaled feature data.
        y (Series): Target variable.

    Returns:
        tuple: Training, validation, and testing sets for features and target variable.
    """
    # Initial split to get a train set and a temporary set for validation/testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # Split the temporary set further into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Train and evaluate Random Forest model
def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Random Forest Regressor model.

    Parameters:
        X_train (ndarray): Training features.
        y_train (Series): Training target variable.
        X_val (ndarray): Validation features.
        y_val (Series): Validation target variable.

    Returns:
        RandomForestRegressor: The trained Random Forest model.
        float: Mean Absolute Error on validation set.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    val_predictions = rf_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print(f"Random Forest Validation MAE: {val_mae:.4f}")

    return rf_model, val_mae


# Train and evaluate Linear Regression model
def train_linear_regression(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Linear Regression model.

    Parameters:
        X_train (ndarray): Training features.
        y_train (Series): Training target variable.
        X_val (ndarray): Validation features.
        y_val (Series): Validation target variable.

    Returns:
        LinearRegression: The trained Linear Regression model.
        float: Mean Absolute Error on validation set.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    val_predictions = lr_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print(f"Linear Regression Validation MAE: {val_mae:.4f}")

    return lr_model, val_mae


# Evaluate the model on the test data
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print the results.

    Parameters:
        model: The trained model (RandomForestRegressor or LinearRegression).
        X_test (ndarray): Test features.
        y_test (Series): Test target variable.

    Returns:
        float: Mean Absolute Error of the model on the test set.
    """
    test_predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    print(f"Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}")
    return test_mae, test_mse


# Plot validation MAE comparison
def plot_model_comparison(rf_mae, lr_mae):
    """
    Plot the validation MAE comparison between Random Forest and Linear Regression.

    Parameters:
        rf_mae (float): Random Forest Validation MAE.
        lr_mae (float): Linear Regression Validation MAE.
    """
    models = ['Random Forest', 'Linear Regression']
    mae_scores = [rf_mae, lr_mae]
    plt.figure(figsize=(8, 4))
    plt.bar(models, mae_scores, color=['skyblue', 'salmon'])
    plt.title('Validation MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    plt.show()


# Main function to execute the entire workflow
def main(filepath):
    """
    Main function to execute data loading, preprocessing, model training, and evaluation.

    Parameters:
        filepath (str): Path to the CSV file containing the dataset.
    """
    # Load and preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train Random Forest model and get validation MAE
    rf_model, rf_mae = train_random_forest(X_train, y_train, X_val, y_val)

    # Train Linear Regression model and get validation MAE
    lr_model, lr_mae = train_linear_regression(X_train, y_train, X_val, y_val)

    # Compare models on validation data
    plot_model_comparison(rf_mae, lr_mae)

    # Evaluate both models on the test data
    print("\nRandom Forest Model Evaluation on Test Data:")
    evaluate_model(rf_model, X_test, y_test)

    print("\nLinear Regression Model Evaluation on Test Data:")
    evaluate_model(lr_model, X_test, y_test)


# Run the app
if __name__ == "__main__":
    main('../data/diabete.csv')
