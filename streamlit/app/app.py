import streamlit as st
from streamlit.modelised_classes.diabetes_prediction_ml_module import DiabetesPredictionML
from streamlit.modelised_classes.diabetes_prediction_module import DiabetesPredictionApp

st.title("Application de Data Science")

# Sidebar Menu
page = st.sidebar.selectbox("Choisissez une page:", ["Accueil", "Régression", "Classification (Deep Learning)", "Classification (ML)"])

if page == "Accueil":
    st.write("Bienvenue sur l'application de Data Science!")
    st.markdown("""
    Naviguez à travers les pages pour explorer :
    - **Régression** : Cas d'utilisation avec des données continues.
    - **Classification (Deep Learning)** : Modèles de réseaux de neurones pour la classification.
    - **Classification (ML)** : Modèles de Machine Learning pour la classification.
    """)

elif page == "Classification (Deep Learning)":
    st.write("## Classification du diabète avec Deep Learning")
    filepath = st.text_input("Entrez le chemin vers le dataset :", value="data/diabete.csv")

    if st.button("Exécuter"):
        if filepath:
            diabetes_app = DiabetesPredictionApp(filepath)
            diabetes_app.execute_workflow()
            st.success("Workflow terminé avec succès ! Modèle sauvegardé.")
        else:
            st.error("Veuillez entrer un chemin valide pour le dataset.")

elif page == "Classification (ML)":
    st.write("## Classification du diabète avec Machine Learning")
    filepath = st.text_input("Entrez le chemin vers le dataset :", value="data/diabete.csv")

    if st.button("Exécuter"):
        if filepath:
            diabetes_ml_app = DiabetesPredictionML(filepath)
            diabetes_ml_app.execute_workflow()
            st.success("Workflow terminé avec succès ! Modèles sauvegardés.")
        else:
            st.error("Veuillez entrer un chemin valide pour le dataset.")
