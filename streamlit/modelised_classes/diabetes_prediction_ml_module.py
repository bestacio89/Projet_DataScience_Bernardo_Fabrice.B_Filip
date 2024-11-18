import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


class DiabetesPredictionML:
    def __init__(self, filepath):
        """
        Initialize the Diabetes Prediction ML utility.

        Parameters:
            filepath (str): Path to the CSV file containing the dataset.
        """
        self.filepath = filepath
        self.scaler = MinMaxScaler()
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.rf_model = None
        self.lr_model = None
        self.rf_mae = None
        self.lr_mae = None

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

    def train_random_forest(self):
        """
        Train and evaluate a Random Forest Regressor model.
        """
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        val_predictions = rf_model.predict(self.X_val)
        self.rf_mae = mean_absolute_error(self.y_val, val_predictions)
        self.rf_model = rf_model
        print(f"Random Forest Validation MAE: {self.rf_mae:.4f}")

    def train_linear_regression(self):
        """
        Train and evaluate a Linear Regression model.
        """
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        val_predictions = lr_model.predict(self.X_val)
        self.lr_mae = mean_absolute_error(self.y_val, val_predictions)
        self.lr_model = lr_model
        print(f"Linear Regression Validation MAE: {self.lr_mae:.4f}")

    def evaluate_model(self, model):
        """
        Evaluate the model on the test set and return metrics.
        """
        test_predictions = model.predict(self.X_test)
        test_mae = mean_absolute_error(self.y_test, test_predictions)
        test_mse = mean_squared_error(self.y_test, test_predictions)
        print(f"Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}")
        return test_mae, test_mse

    def save_model(self, model, directory, filename):
        """
        Save the trained model to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, filename)
        joblib.dump(model, save_path)
        print(f"Model saved at {save_path}")

    def plot_model_comparison(self):
        """
        Plot the validation MAE comparison between Random Forest and Linear Regression.
        """
        models = ['Random Forest', 'Linear Regression']
        mae_scores = [self.rf_mae, self.lr_mae]
        plt.figure(figsize=(8, 4))
        plt.bar(models, mae_scores, color=['skyblue', 'salmon'])
        plt.title('Validation MAE Comparison')
        plt.ylabel('Mean Absolute Error')
        plt.show()

    def execute_workflow(self):
        """
        Execute the entire workflow: preprocessing, training, and evaluation.
        """
        X, y = self.load_and_preprocess_data()
        self.split_data(X, y)
        self.train_random_forest()
        self.train_linear_regression()
        self.plot_model_comparison()
        print("\nRandom Forest Model Evaluation on Test Data:")
        self.evaluate_model(self.rf_model)
        print("\nLinear Regression Model Evaluation on Test Data:")
        self.evaluate_model(self.lr_model)
        self.save_model(self.rf_model, "model/diabete", "DiabeteRandomForestModel.pkl")
        self.save_model(self.lr_model, "model/diabete", "DiabeteLinearRegressionModel.pkl")
