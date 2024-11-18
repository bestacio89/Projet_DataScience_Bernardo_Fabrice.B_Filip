"""
Diabetes Prediction App using a Deep Learning Model in TensorFlow

This script trains a neural network model to predict diabetes-related outcomes
using a provided dataset. The script includes data preprocessing, model training,
evaluation, hyperparameter tuning, and architecture optimization.

Author: Bernardo Estacio Abreu, Fabrice Bellin, Filip Dabrowsky
Date: 15/11/2024
"""


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Detect numerical and categorical columns
def detect_columns(data):
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical_columns, numerical_columns

# Preprocess categorical data
def preprocess_categorical_data(data, categorical_columns, strategy, encoding_method):
    if strategy == "Fill with mode":
        for col in categorical_columns:
            data[col].fillna(data[col].mode()[0], inplace=True)
    elif strategy == "Drop rows":
        data.dropna(subset=categorical_columns, inplace=True)

    if encoding_method == "Label Encoding":
        for col in categorical_columns:
            data[col] = data[col].astype("category").cat.codes
    elif encoding_method == "One-Hot Encoding":
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data

# Preprocess numerical data
def preprocess_numerical_data(data, numerical_columns, strategy):
    if strategy == "Fill with mean":
        for col in numerical_columns:
            data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == "Drop rows":
        data.dropna(subset=numerical_columns, inplace=True)

    return data

# Plot data distributions
def plot_distributions(data, numerical_columns):
    st.write("### Data Distributions")
    for col in numerical_columns:
        try:
            fig, ax = plt.subplots()
            data[col].hist(ax=ax, bins=15, color="skyblue", edgecolor="black")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot distribution for {col}: {e}")

# Plot correlation heatmap
def plot_correlation_heatmap(data):
    try:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate correlation heatmap: {e}")

# Streamlit App
st.title("Interactive ML Application with Error Handling")

# Tabs
tab1, tab2, tab3 = st.tabs(["Data & Model", "Graphs & Analysis", "Hyperparameter Tuning"])

# File uploader and preprocessing in the first tab
with tab1:
    st.write("## Data & Model")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(data)

            # Detect numerical and categorical columns
            categorical_columns, numerical_columns = detect_columns(data)
            st.write("### Categorical Columns Detected")
            st.write(categorical_columns)
            st.write("### Numerical Columns Detected")
            st.write(numerical_columns)

            # Handle missing values
            st.write("### Handle Missing Values")
            cat_missing_strategy = st.selectbox(
                "Choose how to handle missing values for categorical columns:",
                ["Fill with mode", "Drop rows", "None"]
            )
            num_missing_strategy = st.selectbox(
                "Choose how to handle missing values for numerical columns:",
                ["Fill with mean", "Drop rows", "None"]
            )

            # Encode categorical data
            encoding_method = st.selectbox("Choose encoding method for categorical data:", ["Label Encoding", "One-Hot Encoding"])

            # Preprocess categorical and numerical data
            if categorical_columns:
                data = preprocess_categorical_data(data, categorical_columns, cat_missing_strategy, encoding_method)
            if numerical_columns:
                data = preprocess_numerical_data(data, numerical_columns, num_missing_strategy)

            st.write("### Preprocessed Dataset")
            st.dataframe(data)

            # Feature and target selection
            target_column = st.selectbox("Select the target column:", data.columns)
            feature_columns = st.multiselect("Select feature columns:", [col for col in data.columns if col != target_column])

            if feature_columns and target_column:
                X = data[feature_columns]
                y = data[target_column]

                # Re-check numerical columns after encoding
                categorical_columns, numerical_columns = detect_columns(X)

                # Scale numerical features
                scaler = StandardScaler()
                if numerical_columns:
                    numerical_columns_in_X = [col for col in numerical_columns if col in X.columns]
                    if numerical_columns_in_X:
                        X[numerical_columns_in_X] = scaler.fit_transform(X[numerical_columns_in_X])

                # Split data
                test_size = st.slider("Test size (proportion):", 0.1, 0.5, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Model selection
                model_choice = st.selectbox("Choose a model:", ["Random Forest", "Perceptron", "Regression"])

                # Initialize model
                model = None
                if model_choice == "Random Forest":
                    n_estimators = st.slider("Number of trees (n_estimators):", 10, 200, 100)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                elif model_choice == "Perceptron":
                    max_iter = st.slider("Maximum Iterations:", 100, 1000, 500)
                    model = Perceptron(max_iter=max_iter, random_state=42)
                elif model_choice == "Regression":
                    model = LinearRegression()

                # Train and evaluate
                if st.button("Train and Evaluate"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if model_choice in ["Random Forest", "Perceptron"]:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"### Accuracy: {accuracy:.2f}")
                        st.write("### Classification Report")
                        st.text(classification_report(y_test, y_pred))

                    elif model_choice == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        st.write(f"### Mean Squared Error: {mse:.2f}")
                        st.write("### Actual vs Predicted")
                        comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head(20)
                        st.write(comparison)
        except Exception as e:
            st.error(f"An error occurred while processing the data: {e}")
    else:
        st.info("Please upload a CSV file to begin.")

# Graphs and analysis in the second tab
with tab2:
    st.write("## Graphs & Analysis")
    if uploaded_file is not None and numerical_columns:
        plot_distributions(data, numerical_columns)
        plot_correlation_heatmap(data)
    else:
        st.info("Please upload a dataset with numerical columns for analysis.")

# Hyperparameter tuning in the third tab
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning in the third tab
with tab3:
    st.write("## Hyperparameter Tuning")

    if 'model' in locals() and uploaded_file is not None:
        # Display options for the selected model
        if model_choice == "Random Forest":
            st.write("### Hyperparameters for Random Forest")
            param_grid = {
                'n_estimators': st.slider("Number of trees (n_estimators):", 10, 200, (50, 150)),
                'max_depth': st.slider("Max Depth:", 2, 20, (5, 15)),
                'min_samples_split': st.slider("Min Samples Split:", 2, 10, (2, 5)),
            }
        elif model_choice == "Perceptron":
            st.write("### Hyperparameters for Perceptron")
            param_grid = {
                'penalty': st.selectbox("Penalty:", [None, 'l2', 'l1', 'elasticnet']),
                'alpha': st.slider("Alpha (Regularization strength):", 0.0001, 1.0, (0.0001, 0.1)),
                'max_iter': st.slider("Maximum Iterations:", 100, 1000, (500, 1000)),
            }
        elif model_choice == "Regression":
            st.info("Regression models are less commonly tuned with grid search. Consider manual adjustments.")
            param_grid = {}

        # Run Grid Search
        if st.button("Run Hyperparameter Tuning"):
            try:
                if param_grid:
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
                    grid_search.fit(X_train, y_train)

                    st.write("### Best Hyperparameters")
                    st.json(grid_search.best_params_)

                    # Evaluate best model
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)

                    if model_choice in ["Random Forest", "Perceptron"]:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"### Accuracy with Best Parameters: {accuracy:.2f}")
                        st.write("### Classification Report")
                        st.text(classification_report(y_test, y_pred))
                    elif model_choice == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        st.write(f"### Mean Squared Error with Best Parameters: {mse:.2f}")
                else:
                    st.warning("No parameters to tune for the selected model.")
            except Exception as e:
                st.error(f"An error occurred during hyperparameter tuning: {e}")
    else:
        st.info("Please upload a dataset and complete model training in the 'Data & Model' tab first.")
