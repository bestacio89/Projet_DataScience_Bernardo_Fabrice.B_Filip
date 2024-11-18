import streamlit as st
import pandas as pd
from src.utils import charger_donnees, preprocessor_donnees
from src.models import charger_modele, entrainer_modele_regression
from src.visualization import afficher_histogrammes, afficher_nuage_points

# Charger les données
data = charger_donnees("data/processed/vin.csv")

# Titre de l'application
st.title("Régression sur la qualité du vin")

# Afficher les données
st.write("Données du vin:")
st.dataframe(data)

# Visualisation des données
st.subheader("Visualisation des données")
afficher_histogrammes(data, ['fixed acidity', 'volatile acidity'])
afficher_nuage_points(data, 'fixed acidity', 'volatile acidity')

# Entraînement du modèle
st.subheader("Entraînement du modèle")
if st.button("Entraîner le modèle"):
    # Préparer les données pour l'entraînement
    X = data.drop('target', axis=1)
    y = data['target']
    X_preprocessed = preprocessor_donnees(X, ['fixed acidity', 'volatile acidity'], ['type'])
    # Entraîner le modèle
    modele = entrainer_modele_regression(X_preprocessed, y)
    # Afficher les coefficients du modèle
    st.write("Coefficients du modèle:")
    st.write(modele.coef_)

# Charger un modèle existant
st.subheader("Charger un modèle existant")
uploaded_file = st.file_uploader("Charger un fichier de modèle (.pkl)", type="pkl")
if uploaded_file is not None:
    modele = charger_modele(uploaded_file)
    st.write("Modèle chargé avec succès!")

# Faire des prédictions
st.subheader("Faire des prédictions")
if st.button("Prédire"):
    # Sélectionner les variables pour la prédiction
    st.write("Sélectionnez les valeurs des variables pour la prédiction :")
    fixed_acidity = st.number_input("Acidité fixe :", value=data['fixed acidity'].mean())
    volatile_acidity = st.number_input("Acidité volatile :", value=data['volatile acidity'].mean())
    # ... ajouter d'autres variables si nécessaire ...

    # Créer un DataFrame avec les valeurs des variables
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        # ... ajouter d'autres variables si nécessaire ...
    })

    # Prétraiter les données d'entrée
    input_data_preprocessed = preprocessor_donnees(input_data, ['fixed acidity', 'volatile acidity'], ['type'])

    # Faire la prédiction
    prediction = modele.predict(input_data_preprocessed)

    # Afficher la prédiction
    st.write(f"Prédiction de la qualité du vin : {prediction[0]:.2f}")