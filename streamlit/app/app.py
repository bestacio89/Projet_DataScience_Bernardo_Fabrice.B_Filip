import streamlit as st

st.title("Application de Data Science")

# Menu
page = st.sidebar.selectbox("Choisissez une page:", ["Accueil", "Régression", "Classification", "Ongles"])

if page == "Accueil":
  st.write("Bienvenue sur l'application de Data Science!")
  # ... (Ajouter du contenu à la page d'accueil)

elif page == "Régression":
  # Exécuter le script de régression
  st.write("## Régression sur la qualité du vin")
  # ... (Code pour la régression, similaire à l'exemple précédent)

elif page == "Classification":
  # Exécuter le script de classification
  st.write("## Classification du diabète")
  # ... (Code pour la classification, similaire à l'exemple précédent)

elif page == "Ongles":
  # Exécuter le script d'analyse d'ongles
  st.write("## Analyse d'images d'ongles")
  # ... (Code pour l'analyse d'ongles, similaire à l'exemple précédent)