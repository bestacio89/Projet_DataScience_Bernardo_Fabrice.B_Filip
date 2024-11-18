import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])  # Drop unnecessary column

    if "target" not in data.columns:
        raise ValueError("The dataset must contain a 'target' column for predictions.")

    X = data.drop(columns=["target"])
    y = data["target"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, layers, dropout_rate, activation):
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


def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, patience):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=1)

    return history


def save_model(model, directory, filename):
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)
    model.save(save_path)
    print(f"Model saved at {save_path}")


def main(filepath):
    model_path = "model/diabete/DiabetePerceptron.keras"

    if os.path.exists(model_path):
        print("Model already exists. Loading the trained model...")
        model = load_model(model_path)
    else:
        print("No pre-trained model found. Starting training process...")

        X, y = load_and_preprocess_data(filepath)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Use fixed parameters to limit retraining
        layers = [128, 64]
        dropout_rate = 0.3
        activation = "relu"
        batch_size = 32
        epochs = 50  # Limit training to a maximum of 50 epochs
        patience = 5  # Stop early if no improvement for 5 epochs

        model = build_model(X_train.shape[1], layers, dropout_rate, activation)
        train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, patience)

        # Save the trained model
        save_model(model, "mode/diabete", "DiabetePerceptron.keras")

    # Evaluate the model on the test set
    X, y = load_and_preprocess_data(filepath)
    _, _, X_test, _, _, y_test = split_data(X, y)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")


if __name__ == "__main__":
    main('../data/diabete.csv')
