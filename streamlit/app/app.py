import streamlit as st
from streamlit.modelised_classes.diabetes_prediction_module import DiabetesPredictionApp

st.title("Application de Data Science")

# Sidebar Menu
page = st.sidebar.selectbox("Choisissez une page:", ["Accueil", "Régression", "Classification", "Ongles"])

if page == "Accueil":
    st.write("Bienvenue sur l'application de Data Science!")
    st.write("Naviguez à travers les pages pour explorer des cas d'utilisation spécifiques :")
    st.markdown("""
    - **Régression** : Modèles de prédiction sur des données continues.
    - **Classification** : Modèles de prédiction binaire ou multi-classes.
    - **Ongles** : Analyse d'images et classification.
    """)

elif page == "Régression":
    st.write("## Régression sur la qualité du vin")
    st.write("Cette section est en cours de développement. Revenez bientôt!")

elif page == "Classification":
    st.write("## Classification du diabète")

    # User input for file path
    filepath = st.text_input("Entrez le chemin vers le dataset du diabète :", value="data/diabete.csv")

    if st.button("Exécuter la classification"):
        if filepath:
            try:
                # Run the DiabetesPrediction workflow
                diabetes_app = DiabetesPredictionApp(filepath)
                diabetes_app.execute_workflow()
                st.success(
                    "Workflow terminé avec succès! Modèle enregistré dans `model/diabete/DiabetePerceptron.keras`")
            except Exception as e:
                st.error(f"Erreur lors de l'exécution : {e}")
        else:
            st.error("Veuillez entrer un chemin valide pour le fichier dataset.")

elif page == "Ongles":
    st.write("## Analyse d'images d'ongles")
    st.write("Cette section est en cours de développement. Revenez bientôt!")
