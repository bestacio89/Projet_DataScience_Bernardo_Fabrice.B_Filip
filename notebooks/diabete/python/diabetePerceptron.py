# diabetes_prediction_app.py

"""
Diabetes Prediction App using a Deep Learning Model in TensorFlow

This script trains a neural network model to predict diabetes-related outcomes
using a provided dataset. The script includes data preprocessing, model training,
evaluation, hyperparameter tuning, and architecture optimization.

Author: Bernardo Estacio Abreu, Fabrice Bellin, Filip Dabrowsky
Date: 15/11/2024
"""

# Import necessary libraries
import os
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
    data = pd.read_csv(filepath)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")  # Drop unnecessary column

    X = data.drop(columns=["target"])
    y = data["target"]

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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# Define and build a neural network model
def build_model(input_shape, layers, dropout_rate, activation):
    """
    Build and compile a neural network model using TensorFlow Keras.

    Parameters:
        input_shape (int): Number of input features.
        layers (list): List containing number of neurons in each hidden layer.
        dropout_rate (float): Dropout rate to prevent overfitting.
        activation (str): Activation function for hidden layers.

    Returns:
        Sequential: A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Dense(layers[0], activation=activation, input_shape=(input_shape,)))

    for neurons in layers[1:]:
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model


# Train the model
def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, patience):
    """
    Train the neural network model with early stopping and given hyperparameters.

    Parameters:
        model (Sequential): Compiled Keras model.
        X_train (ndarray): Training features.
        y_train (Series): Training target variable.
        X_val (ndarray): Validation features.
        y_val (Series): Validation target variable.
        batch_size (int): Number of samples per batch.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs with no improvement to stop training.

    Returns:
        History: Training history object.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=1)

    return history


# Hyperparameter tuning function
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Tune hyperparameters by testing multiple configurations.

    Parameters:
        X_train (ndarray): Training features.
        y_train (Series): Training target variable.
        X_val (ndarray): Validation features.
        y_val (Series): Validation target variable.

    Returns:
        Sequential: Best Keras model based on validation performance.
        dict: Best hyperparameters.
    """
    best_mae = float("inf")
    best_model = None
    best_params = {}

    # Hyperparameter grid
    layer_configs = [[128, 64], [256, 128, 64]]
    dropout_rates = [0.2, 0.3, 0.4]
    activations = ['relu', 'tanh']
    batch_sizes = [16, 32]
    epochs = 200
    patience = 15

    # Grid search
    for layers in layer_configs:
        for dropout_rate in dropout_rates:
            for activation in activations:
                for batch_size in batch_sizes:
                    print(f"Testing: Layers={layers}, Dropout={dropout_rate}, Activation={activation}, Batch Size={batch_size}")
                    model = build_model(X_train.shape[1], layers, dropout_rate, activation)
                    history = train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, patience)
                    val_mae = min(history.history['val_mean_absolute_error'])

                    if val_mae < best_mae:
                        best_mae = val_mae
                        best_model = model
                        best_params = {
                            'layers': layers,
                            'dropout_rate': dropout_rate,
                            'activation': activation,
                            'batch_size': batch_size,
                            'patience': patience
                        }

    print(f"Best Hyperparameters: {best_params}")
    return best_model, best_params


# Save the model
def save_model(model, directory, filename):
    """
    Save the trained model to the specified directory.

    Parameters:
        model (Sequential): Trained Keras model.
        directory (str): Directory to save the model.
        filename (str): Filename for the saved model.
    """
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)
    model.save(save_path)
    print(f"Model saved at {save_path}")


# Main function
def main(filepath):
    """
    Main function to execute the entire workflow.

    Parameters:
        filepath (str): Path to the CSV file containing the dataset.
    """
    X, y = load_and_preprocess_data(filepath)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Tune hyperparameters and get the best model
    best_model, best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    loss, mae = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # Save the best model
    save_model(best_model, "model/diabete", "DiabetePerceptron.keras")


# Run the script
if __name__ == "__main__":
    main('../../data/diabete.csv')
