import streamlit as st
import pandas as pd

# ... (ton code pour les imports et les fonctions)

st.title("Application de Data Science")

# Menu
page = st.sidebar.selectbox("Choisissez une page:", ["Accueil", "Régression", "Classification", "Ongles"])

if page == "Accueil":
    st.header("Bienvenue !")
    st.write(
        """
        Explorez et analysez des données sur le vin et le diabète,
        entraînez des modèles de Machine Learning et faites des prédictions.
        """
    )

    st.subheader("Fonctionnalités :")
    st.markdown(
        """
        * **Régression :** Prédire la qualité du vin.
        * **Classification :** Prédire le diabète.
        * **Ongles :** Analyser des images d'ongles (à venir).
        """
    )

    # Image en largeur 100% (chemin corrigé)
    st.image("data/docteur.jpg", use_container_width=True)  # Chemin adapté

# ... (ton code pour les autres pages)

if page == "Régression":
    st.header("Bienvenue !")
    choix_algorythme = st.selectbox("Selectionnez", ["1", "2", "3"])
    st.write(choix_algorythme)