# diabetes_prediction_app.py

"""
Diabetes Prediction App using a Deep Learning Model in TensorFlow

This script trains a neural network model to predict diabetes-related outcomes
using a provided dataset. The script includes data preprocessing, model training,
evaluation, and optional hyperparameter tuning.

Author: Bernardo Estacio Abrei, Fabrice Bellin, Filip Dabrowsky
Date: 15/11/2024
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


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


# Build the neural network model
def build_model(input_shape):
    """
    Build and compile a neural network model using TensorFlow Keras.

    Parameters:
        input_shape (int): The number of features in the input data.

    Returns:
        Sequential: A compiled Keras Sequential model.
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Output layer for regression
    ])

    # Compile model with mean squared error as loss for regression
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model


# Train the model with early stopping
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the neural network model with early stopping to avoid overfitting.

    Parameters:
        model (Sequential): The compiled neural network model.
        X_train (ndarray): Training features.
        y_train (Series): Training target variable.
        X_val (ndarray): Validation features.
        y_val (Series): Validation target variable.

    Returns:
        History: Training history object with details on training and validation loss.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Fit the model with training data and validate on validation set
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=200,
                        batch_size=16,
                        callbacks=[early_stopping],
                        verbose=1)

    return history


# Evaluate the model on the test data
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print the results.

    Parameters:
        model (Sequential): The trained neural network model.
        X_test (ndarray): Test features.
        y_test (Series): Test target variable.

    Returns:
        float: Mean Absolute Error of the model on the test set.
    """
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
    return mae


# Plot training and validation loss
def plot_training_history(history):
    """
    Plot the training and validation loss over epochs.

    Parameters:
        history (History): Training history returned by model.fit.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
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

    # Build model
    model = build_model(X_train.shape[1])

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot training history
    plot_training_history(history)


# Run the app
if __name__ == "__main__":
    main('../data/diabete.csv')
