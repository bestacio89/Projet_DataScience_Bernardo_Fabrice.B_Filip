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
        fig, ax = plt.subplots()
        data[col].hist(ax=ax, bins=15, color="skyblue", edgecolor="black")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Plot correlation heatmap
def plot_correlation_heatmap(data):
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Streamlit App
st.title("Interactive ML Application with Hyperparameter Tuning and Visualization")

# Tabs
tab1, tab2, tab3 = st.tabs(["Data & Model", "Graphs & Analysis", "Hyperparameter Tuning"])

# File uploader and preprocessing in the first tab
with tab1:
    st.write("## Data & Model")
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

# Graphs and analysis in the second tab
with tab2:
    if uploaded_file is not None:
        st.write("## Graphs & Analysis")
        if numerical_columns:
            plot_distributions(data, numerical_columns)
            plot_correlation_heatmap(data)
        else:
            st.write("No numerical columns available for analysis.")

# Hyperparameter tuning in the third tab
with tab3:
    st.write("## Hyperparameter Tuning")
    if model_choice == "Random Forest":
        st.write("### Random Forest Hyperparameters")
        n_estimators = st.slider("Number of Trees (n_estimators):", 10, 300, n_estimators)
        max_depth = st.slider("Maximum Depth:", 1, 50, 10)
        model.set_params(n_estimators=n_estimators, max_depth=max_depth)
    elif model_choice == "Perceptron":
        st.write("### Perceptron Hyperparameters")
        max_iter = st.slider("Maximum Iterations:", 100, 2000, max_iter)
        alpha = st.slider("Alpha (Regularization Term):", 0.0001, 0.1, 0.001, step=0.0001)
        model.set_params(max_iter=max_iter, alpha=alpha)

    if st.button("Re-Train with Tuned Parameters"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_choice in ["Random Forest", "Perceptron"]:
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Tuned Accuracy: {accuracy:.2f}")
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

        elif model_choice == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"### Tuned Mean Squared Error: {mse:.2f}")
