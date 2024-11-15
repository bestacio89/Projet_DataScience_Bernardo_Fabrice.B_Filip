import streamlit as st
from PIL import Image
from src.models import analyser_image_ongle

st.title("Analyse d'images d'ongles")

uploaded_file = st.file_uploader("Charger une image d'ongle", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Image chargée", use_column_width=True)

  if st.button("Analyser l'image"):
    analyse = analyser_image_ongle(image)
    st.write("Résultats de l'analyse:")
    st.write(analyse)