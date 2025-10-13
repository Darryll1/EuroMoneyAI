import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import os
import time

# Mapping classes
int_to_class = {0:"10c",1:"1c",2:"1e",3:"20c",4:"2c",5:"2e",6:"50c",7:"5c"}

# Créer dossier local pour les poids
os.makedirs("saved_models", exist_ok=True)

# Lien Google Drive direct vers tes poids du Dense + Dropout
WEIGHTS_URL = "https://drive.google.com/uc?id=17r6xrf0vM7wNMrPSaGv8xqUV0cuOl6OE"
WEIGHTS_PATH = "saved_models/model_weights_final.weights.h5"

# Télécharger le fichier si absent
if not os.path.exists(WEIGHTS_PATH):
    r = requests.get(WEIGHTS_URL, stream=True)
    r.raise_for_status()
    with open(WEIGHTS_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

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
best_model.load_weights(WEIGHTS_PATH)

# Interface Streamlit
st.title("Prédicteur de pièces d'euro")
uploaded_file = st.file_uploader("Choisis une image de pièce", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)
    img_array = img_to_array(image.resize((150,150)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    start = time.time()
    preds = best_model.predict(img_array)
    elapsed = time.time() - start

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)*100

    st.subheader(f"Valeur prédite : {int_to_class[predicted_class]}")
    st.metric("Confiance du modèle", f"{confidence:.2f} %")
    st.metric("Temps de prédiction", f"{elapsed:.3f} sec")
