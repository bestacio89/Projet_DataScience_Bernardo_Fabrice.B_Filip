import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

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

# Streamlit App
st.title("Mixed Data Handling App with Fine-Tunable Machine Learning Models")
st.write("Upload, preprocess, and apply machine learning models to predict the target label.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
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
    st.write("### Modeling")
    target_column = st.selectbox("Select the target column:", data.columns)
    feature_columns = st.multiselect("Select feature columns:", [col for col in data.columns if col != target_column])

    if feature_columns and target_column:
        X = data[feature_columns]
        y = data[target_column]

        # Scale numerical features
        scaler = StandardScaler()
        if numerical_columns:
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

        # Split data
        test_size = st.slider("Test size (proportion):", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model selection
        model_choice = st.selectbox("Choose a model:", ["Random Forest", "Perceptron", "Regression"])

        # Initialize model with default parameters
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

            # Fine-tuning options
            st.write("### Fine-Tuning Options")
            if model_choice == "Random Forest":
                n_estimators = st.slider("Fine-tune Number of Trees (n_estimators):", 10, 300, n_estimators)
                model.set_params(n_estimators=n_estimators)
            elif model_choice == "Perceptron":
                max_iter = st.slider("Fine-tune Maximum Iterations:", 100, 2000, max_iter)
                model.set_params(max_iter=max_iter)

            if st.button("Re-Train with Fine-Tuned Parameters"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if model_choice in ["Random Forest", "Perceptron"]:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"### Fine-Tuned Accuracy: {accuracy:.2f}")
                    st.write("### Classification Report")
                    st.text(classification_report(y_test, y_pred))

                elif model_choice == "Regression":
                    mse = mean_squared_error(y_test, y_pred)
                    st.write(f"### Fine-Tuned Mean Squared Error: {mse:.2f}")
else:
    st.write("Please upload a CSV file to begin.")
