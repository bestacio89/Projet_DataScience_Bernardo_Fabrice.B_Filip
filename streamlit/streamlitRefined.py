import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Streamlit Page Configuration
st.set_page_config(
    page_title="Deep Learning - Academic Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar with project details
st.sidebar.title("About This Project")
st.sidebar.info(
    """
    **Deep Learning Academic Project**
    - **Authors**: Bernardo Estacio Abreu, Fabrice Bellin, Filip Dabrowsky
    - **Date**: November 2024
    - **Objective**: Predict diabetes outcomes using machine learning models.

    For more details, visit [GitHub Repository](https://github.com/your-repo).
    """
)

# Academic Header with Markdown Styling
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #4CAF50; font-family: 'Times New Roman';">
            Deep Learning Academic Project
        </h1>
        <p style="font-size: 18px; font-family: 'Arial';">
           Ce projet utilise des techniques d'apprentissage automatique pour prédire les résultats liés au diabète.
           Les modèles explorés incluent Perceptron, Forêt Aléatoire et Régression Linéaire.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Paths to Models
PERCETRON_MODEL_PATH = "model/diabete/DiabetePerceptron.keras"
RANDOM_FOREST_MODEL_PATH = "model/diabete/DiabeteRandomForestModel.pkl"
LINEAR_REGRESSION_MODEL_PATH = "model/diabete/DiabeteLinearRegressionModel.pkl"

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_val' not in st.session_state:
    st.session_state['X_val'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_val' not in st.session_state:
    st.session_state['y_val'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'feature_columns' not in st.session_state:
    st.session_state['feature_columns'] = None

# Tabs
tab1, tab2, tab3 = st.tabs(["Preprocessing", "Charts and Plots", "Model Training & Fine-Tuning"])

# Helper Functions
def load_and_preprocess_data(file):
    """Load and preprocess dataset."""
    data = pd.read_csv(file)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data

def scale_features(X):
    """Scale features using MinMaxScaler."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_data(X, y):
    """Split data into train, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Tab 1: Preprocessing
with tab1:
    st.header("Preprocessing")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file:
        data = load_and_preprocess_data(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Select Target Column
        target_column = st.selectbox("Select the target column:", data.columns)
        feature_columns = [col for col in data.columns if col != target_column]

        # Scale Features
        X = data[feature_columns]
        y = data[target_column]
        X_scaled = scale_features(X)

        # Split Data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

        # Save to session state
        st.session_state['data'] = data
        st.session_state['feature_columns'] = feature_columns
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
        st.session_state['y_test'] = y_test

        st.success("Data preprocessed and split successfully!")

# Tab 2: Charts and Plots
with tab2:
    st.header("Charts and Plots")

    if st.session_state['data'] is not None:
        data = st.session_state['data']
        feature_columns = st.session_state['feature_columns']

        st.write("### Data Distributions")
        for col in feature_columns:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, color="skyblue", ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[feature_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please upload and preprocess data in the Preprocessing tab.")

# Tab 3: Model Training & Fine-Tuning
with tab3:
    st.header("Model Training & Fine-Tuning")

    if st.session_state['X_train'] is not None:
        X_train = st.session_state['X_train']
        X_val = st.session_state['X_val']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_val = st.session_state['y_val']
        y_test = st.session_state['y_test']

        # Model Selection
        model_choice = st.selectbox("Select a Model", ["Perceptron", "Random Forest", "Linear Regression"])

        # Perceptron (Neural Network)
        if model_choice == "Perceptron":
            st.subheader("Perceptron (Neural Network)")

            # Load the pre-trained model if it exists
            if os.path.exists(PERCETRON_MODEL_PATH):
                model = load_model(PERCETRON_MODEL_PATH)
                st.success("Loaded pre-trained Perceptron model.")
            else:
                st.warning("No pre-trained Perceptron model found. Training a new one.")

                # Model Configuration
                neurons = st.slider("Number of Neurons (Hidden Layer)", min_value=1, max_value=100, value=32, step=1)
                dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
                epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=50, step=10)
                batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)

                # Train a new model
                model = Sequential([
                    Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
                    Dropout(dropout_rate),
                    Dense(1, activation='linear')
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
                model.save(PERCETRON_MODEL_PATH)
                st.success("Trained and saved a new Perceptron model.")

            # Evaluate the model
            st.write("### Model Performance")
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test MAE: {test_mae:.4f}")

        # Random Forest
        elif model_choice == "Random Forest":
            st.subheader("Random Forest Regressor")

            # Load the pre-trained model if it exists
            if os.path.exists(RANDOM_FOREST_MODEL_PATH):
                rf_model = joblib.load(RANDOM_FOREST_MODEL_PATH)
                st.success("Loaded pre-trained Random Forest model.")
            else:
                st.warning("No pre-trained Random Forest model found. Training a new one.")

                # Model Configuration
                n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
                max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=10, step=1)

                # Train a new model
                rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_model.fit(X_train, y_train)
                joblib.dump(rf_model, RANDOM_FOREST_MODEL_PATH)
                st.success("Trained and saved a new Random Forest model.")

            # Evaluate the model
            st.write("### Model Performance")
            predictions = rf_model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Test MAE: {mae:.4f}")
            st.write(f"Test MSE: {mse:.4f}")

        # Linear Regression
        elif model_choice == "Linear Regression":
            st.subheader("Linear Regression")

            # Load the pre-trained model if it exists
            if os.path.exists(LINEAR_REGRESSION_MODEL_PATH):
                lr_model = joblib.load(LINEAR_REGRESSION_MODEL_PATH)
                st.success("Loaded pre-trained Linear Regression model.")
            else:
                st.warning("No pre-trained Linear Regression model found. Training a new one.")

                # Train a new model
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                joblib.dump(lr_model, LINEAR_REGRESSION_MODEL_PATH)
                st.success("Trained and saved a new Linear Regression model.")

            # Evaluate the model
            st.write("### Model Performance")
            predictions = lr_model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Test MAE: {mae:.4f}")
            st.write(f"Test MSE: {mse:.4f}")

    else:
        st.warning("Please complete preprocessing in the Preprocessing tab.")


# Footer with Markdown Styling
st.markdown(
    """
    ---
    <div style="text-align: center; font-size: 12px; color: gray;">
        <p>© 2024 Deep Learning Academic Project</p>
        <p>Developé par Bernardo Estacio Abreu, Fabrice Bellin, and Filip Dabrowsky</p>
    </div>
    """,
    unsafe_allow_html=True,
)
