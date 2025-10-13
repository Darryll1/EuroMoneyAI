import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import requests
import os
import time

# --- Mapping classes ---
int_to_class = {
    0: "10c", 1: "1c", 2: "1e", 3: "20c", 4: "2c", 5: "2e", 6: "50c", 7: "5c"
}

# --- Créer dossier local ---
os.makedirs("saved_models", exist_ok=True)

# --- Téléchargement depuis Drive ---
ARCHITECTURE_URL = "https://drive.google.com/uc?id=15kAmiCYGfuKqWA6wfsYxaPRwznSxaH-8"
WEIGHTS_URL = "https://drive.google.com/uc?id=1NGqeeIdF_WnVqZg1SFqdbGO6cJw9A4ti"
ARCHITECTURE_PATH = "saved_models/model_architecture.json"
WEIGHTS_PATH = "saved_models/model_weights.weights.h5"

def download_file(url, output_path):
    if not os.path.exists(output_path):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

download_file(ARCHITECTURE_URL, ARCHITECTURE_PATH)
download_file(WEIGHTS_URL, WEIGHTS_PATH)

# --- Charger le modèle ---
with open(ARCHITECTURE_PATH, "r") as f:
    best_model = model_from_json(f.read())

best_model.load_weights(WEIGHTS_PATH)

# --- Interface Streamlit ---
st.set_page_config(page_title="Prédicteur de pièces d'euro", page_icon="€", layout="centered")
st.title("Prédicteur de pièces d'euro")
st.write("Téléverse une image d’une pièce pour obtenir sa valeur prédite.")

uploaded_file = st.file_uploader("Choisis une image de pièce", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image importée", use_container_width=True)

    img_resized = image.resize((150,150))
    img_array = img_to_array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    start_time = time.time()
    preds = best_model.predict(img_array)
    elapsed_time = time.time() - start_time

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)*100

    st.subheader(f"Valeur prédite : **{int_to_class[predicted_class]}**")
    st.metric("Confiance du modèle", f"{confidence:.2f} %")
    st.metric("Temps de prédiction", f"{elapsed_time:.3f} sec")

    st.write("### Probabilités pour chaque classe :")
    for i, cls in int_to_class.items():
        st.write(f"{cls} : {preds[0][i]*100:.2f} %")

    st.bar_chart({int_to_class[i]: float(preds[0][i]*100) for i in range(len(int_to_class))})
else:
    st.info("Téléverse une image pour commencer.")
