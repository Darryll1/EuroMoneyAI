import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import time 
import gdown
import os

st.write("Version de TensorFlow sur cet environnement :", tf.__version__)

int_to_class = {
    0: "10c",
    1: "1c",
    2: "1e",
    3: "20c",
    4: "2c",
    5: "2e",
    6: "50c",
    7: "5c"
}

# Charger le meilleur modèle sauvegardé


url = "https://drive.google.com/uc?id=1PYtp5y6hvn0gBeb2oU1ovVHF3EvFPJUn"
output = "best_model5_final_compatible.h5"
gdown.download(url, output, quiet=False)

best_model = tf.keras.models.load_model("best_model5_final_compatible.h5")


#INTERFACE
st.set_page_config(page_title="Prédicteur de pièces d'euro", page_icon="€", layout="centered")
st.title("Prédicteur de pièces d'euro")
st.write("Téléverse une image d’une pièce pour obtenir sa valeur prédite.")

uploaded_file = st.file_uploader("Choisis une image de pièce", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image importée", use_container_width=True)

    # --- Transformation identique au test_generator ---
    img_resized = image.resize((150, 150))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Chronomètre pour la prédiction ---
    start_time = time.time()
    preds = best_model.predict(img_array)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Classe prédite et confiance
    predicted_classes = np.argmax(preds, axis=1)
    confidence = np.max(preds) * 100

    st.subheader(f"Valeur prédite : **{int_to_class[predicted_classes[0]]}**")
    st.metric(label="Confiance du modèle", value=f"{confidence:.2f} %")
    st.metric(label="Temps de prédiction", value=f"{elapsed_time:.3f} sec")

    # --- Affichage des pourcentages pour toutes les classes ---
    st.write("### Probabilités pour chaque classe :")
    for i, cls in int_to_class.items():
        st.write(f"{cls} : {preds[0][i]*100:.2f} %")

    # --- Bar chart pour visualisation ---
    st.bar_chart({int_to_class[i]: float(preds[0][i]*100) for i in range(len(int_to_class))})

else:
    st.info("Téléverse une image pour commencer.")


