import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import os
import time
import gdown


# Mapping classes
int_to_class = {0:"10c",1:"1c",2:"1e",3:"20c",4:"2c",5:"2e",6:"50c",7:"5c"}

# Créer dossier local pour les poids
os.makedirs("saved_models", exist_ok=True)

# Lien Google Drive direct vers tes poids du Dense + Dropout

file_id = "17r6xrf0vM7wNMrPSaGv8xqUV0cuOl6OE"
output = "saved_models/model_weights_final.weights.h5"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

#WEIGHTS_URL = "https://drive.google.com/uc?id=17r6xrf0vM7wNMrPSaGv8xqUV0cuOl6OE"
#WEIGHTS_PATH = "saved_models/model_weights_final.weights.h5"

# Télécharger le fichier si absent
# Télécharger le fichier si absent
#if not os.path.exists(WEIGHTS_PATH):
#    r = requests.get(WEIGHTS_URL, stream=True)
#    r.raise_for_status()
#    with open(WEIGHTS_PATH, "wb") as f:
#        for chunk in r.iter_content(chunk_size=8192):
#            f.write(chunk)

# Vérifier que le fichier téléchargé est bien un vrai fichier HDF5
if os.path.exists(output):
    size = os.path.getsize(output)
    st.write(f"Fichier téléchargé : {output} ({size/1024:.1f} Ko)")
    with open(output, "rb") as f:
        st.code(f.read(200))  # affiche les 200 premiers octets


# Recréer VGG16 + Dense comme à l'entraînement
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False 
for layer in conv_base.layers:
    if "block5" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

best_dense_units = 256  
best_dropout_rate = 0.3183658163946006

best_model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(best_dense_units, activation='relu'),
    layers.Dropout(best_dropout_rate),
    layers.Dense(8, activation='softmax')
])

# Charger uniquement les poids du Dense + Dropout
best_model.load_weights(output)

# --- INTERFACE ---
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

    st.metric("Temps de prédiction", f"{elapsed:.3f} sec")
