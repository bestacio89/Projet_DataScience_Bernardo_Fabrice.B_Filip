import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


class DiabetesPredictionApp:
    def __init__(self, filepath):
        """
        Initialize the Diabetes Prediction App.

        Parameters:
            filepath (str): Path to the CSV file containing the dataset.
        """
        self.filepath = filepath
        self.scaler = MinMaxScaler()
        self.model = None
        self.best_params = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    def load_and_preprocess_data(self):
        """
        Load the dataset and preprocess it by scaling features.
        """
        data = pd.read_csv(self.filepath)
        data = data.drop(columns=["Unnamed: 0"], errors="ignore")  # Drop unnecessary column

        X = data.drop(columns=["target"])
        y = data["target"]

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def split_data(self, X, y):
        """
        Split the data into training, validation, and testing sets.
        """
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def build_model(self, input_shape, layers, dropout_rate, activation):
        """
        Build and compile a neural network model using TensorFlow Keras.
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

    def train_model(self, model, batch_size, epochs, patience):
        """
        Train the neural network model with early stopping and given hyperparameters.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def tune_hyperparameters(self):
        """
        Tune hyperparameters by testing multiple configurations.
        """
        best_mae = float("inf")
        best_model = None

        # Hyperparameter grid
        layer_configs = [[128, 64], [256, 128, 64]]
        dropout_rates = [0.2, 0.3, 0.4]
        activations = ['relu', 'tanh']
        batch_sizes = [16, 32]
        epochs = 200
        patience = 15

        for layers in layer_configs:
            for dropout_rate in dropout_rates:
                for activation in activations:
                    for batch_size in batch_sizes:
                        print(f"Testing: Layers={layers}, Dropout={dropout_rate}, Activation={activation}, Batch Size={batch_size}")
                        model = self.build_model(self.X_train.shape[1], layers, dropout_rate, activation)
                        history = self.train_model(model, batch_size, epochs, patience)
                        val_mae = min(history.history['val_mean_absolute_error'])

                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_model = model
                            self.best_params = {
                                'layers': layers,
                                'dropout_rate': dropout_rate,
                                'activation': activation,
                                'batch_size': batch_size,
                                'patience': patience
                            }

        self.model = best_model
        print(f"Best Hyperparameters: {self.best_params}")

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
        return loss, mae

    def save_model(self, directory, filename):
        """
        Save the trained model to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, filename)
        self.model.save(save_path)
        print(f"Model saved at {save_path}")

    def run(self):
        """
        Run the entire workflow for training and evaluating the model.
        """
        X, y = self.load_and_preprocess_data()
        self.split_data(X, y)
        self.tune_hyperparameters()
        self.evaluate_model()
        self.save_model("model/diabete", "DiabetePerceptron.keras")
