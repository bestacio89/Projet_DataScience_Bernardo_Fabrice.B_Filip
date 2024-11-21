"""
Diabetes Prediction App using Multiple Machine Learning Models

This script trains various machine learning models to predict diabetes-related outcomes
using a provided dataset. It includes data preprocessing, model training, evaluation,
hyperparameter tuning, and architecture optimization.

Author: Bernardo Estacio Abreu, Fabrice Bellin, Filip Dabrowsky
Date: 21/11/2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocess the data
def preprocess_data(data, method):
    if method == "Fill with mean":
        return data.fillna(data.mean())
    elif method == "Drop rows":
        return data.dropna()
    elif method == "Fill with zero":
        return data.fillna(0)
    else:
        return data

# Plot feature distributions
def plot_feature_distributions(data, feature_columns):
    fig, ax = plt.subplots(figsize=(10, 6))
    data[feature_columns].hist(ax=ax, bins=15, color="skyblue", edgecolor="black")
    st.pyplot(fig)

# Confusion matrix plot
def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Streamlit App
st.title("Diabetes Dataset Machine Learning App with Visualizations")
st.write("Explore, preprocess, and apply multiple machine learning models on your dataset.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Handle missing values
    missing_value_strategy = st.selectbox(
        "Choose how to handle missing values:",
        ["Fill with mean", "Drop rows", "Fill with zero", "None"]
    )
    data = preprocess_data(data, missing_value_strategy)

    # Display dataset summary
    st.write("### Data Summary")
    st.write(data.describe())

    # Feature and target selection
    columns = data.columns.tolist()
    feature_columns = st.multiselect("Select feature columns:", columns, default=columns[:-1])
    target_column = st.selectbox("Select the target column:", columns, index=len(columns) - 1)

    if feature_columns and target_column:
        X = data[feature_columns]
        y = data[target_column]

        # Plot feature distributions
        st.write("### Feature Distributions")
        plot_feature_distributions(data, feature_columns)

        # Split data
        test_size = st.slider("Test size (proportion):", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model selection
        model_choice = st.selectbox(
            "Choose a model:",
            ["Random Forest", "Perceptron", "Logistic Regression", "SVM", "Decision Tree",
             "K-Nearest Neighbors", "Gradient Boosting", "Linear Regression"]
        )

        # Hyperparameter configuration
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of trees (n_estimators):", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        elif model_choice == "Perceptron":
            max_iter = st.slider("Maximum Iterations:", 100, 1000, 500)
            model = Perceptron(max_iter=max_iter, random_state=42)
        elif model_choice == "Logistic Regression":
            C = st.slider("Inverse of regularization strength (C):", 0.01, 10.0, 1.0)
            model = LogisticRegression(C=C, max_iter=1000)
        elif model_choice == "SVM":
            C = st.slider("Regularization parameter (C):", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel type:", ["linear", "rbf", "poly", "sigmoid"])
            model = SVC(C=C, kernel=kernel, random_state=42)
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Maximum depth:", 1, 20, 10)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_choice == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors (k):", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_choice == "Gradient Boosting":
            learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
            n_estimators = st.slider("Number of trees (n_estimators):", 10, 200, 100)
            model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
        elif model_choice == "Linear Regression":
            model = LinearRegression()

        # Train and evaluate
        if st.button("Train and Evaluate"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if model_choice in ["Random Forest", "Perceptron", "Logistic Regression", "SVM", "Decision Tree", "K-Nearest Neighbors", "Gradient Boosting"]:
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"### Accuracy: {accuracy:.2f}")
                st.write("### Classification Report")
                st.text(classification_report(y_test, y_pred))

                # Plot confusion matrix
                st.write("### Confusion Matrix")
                plot_confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

            elif model_choice == "Linear Regression":
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"### Mean Squared Error: {mse:.2f}")
                st.write("### Actual vs Predicted")
                comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head(20)
                st.write(comparison)

else:
    st.write("Please upload a CSV file to begin.")
