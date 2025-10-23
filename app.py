import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os
import time

# --- Configuration page ---
st.set_page_config(
    page_title="BNSM - Sécurité Monétaire",
    page_icon="🏦",
    layout="wide"
)


st.title("Bienvenue à la Banque Nationale de Sécurité Monétaire (BNSM)")


# --- Logo centré ---
col_g, col_c, col_d = st.columns([1, 1, 1])  # symétrique = centre exact
with col_c:
    logo = Image.open("BNSM.png")
    st.image(logo, width=300)  # ajuste la largeur




st.write("Sélectionnez l'option souhaitée ci-dessous pour analyser un billet ou une pièce.")

# --- Menu navigation ---
option = st.selectbox("Que voulez-vous faire ?", 
                      ("","Vérifier l'authenticité d'un billet", "Identifier une pièce d'euro"))
if option == "":
    st.info("Veuillez sélectionner une option pour continuer.")
else:
    # --- Créer dossier local pour les modèles ---
    os.makedirs("saved_models", exist_ok=True)
    
    # ----------------- Section Billet -----------------
    if option == "Vérifier l'authenticité d'un billet":
        st.header("Détection de faux billets")
        st.write("Entrez les caractéristiques du billet :")
        
        # Inputs utilisateur
        diagonal = st.number_input("Diagonal", min_value=0.0, max_value=200.0, value=171.81)
        height_left = st.number_input("Height Left", min_value=0.0, max_value=200.0, value=104.86)
        height_right = st.number_input("Height Right", min_value=0.0, max_value=200.0, value=104.95)
        margin_low = st.number_input("Margin Low", min_value=0.0, max_value=10.0, value=4.52)
        margin_up = st.number_input("Margin Up", min_value=0.0, max_value=10.0, value=2.89)
        length = st.number_input("Length", min_value=0.0, max_value=200.0, value=112.83)

        st.markdown("<div style='margin-bottom: 200px;'></div>", unsafe_allow_html=True)

        
        # Télécharger modèles si pas déjà
        scaler_id = "18JiwJMmIQNxPb08VipBrw0lWcAxDSRaq"
        output_scaler = "saved_models/scaler.pkl"
        if not os.path.exists(output_scaler):
            gdown.download(f"https://drive.google.com/uc?id={scaler_id}", output_scaler, quiet=False)
        
        model_id = "12Wzx3g_9FfnSgSD0GfIy5CiIs92Wx1rk"
        output_model = "saved_models/random_forest_best_model.pkl"
        if not os.path.exists(output_model):
            gdown.download(f"https://drive.google.com/uc?id={model_id}", output_model, quiet=False)
        
        # Charger scaler et modèle
        scaler = joblib.load(output_scaler)
        rf_model = joblib.load(output_model)
        
        if st.button("Vérifier le billet"):
            input_features = np.array([[diagonal, height_left, height_right, margin_low, margin_up, length]])
            input_features_scaled = scaler.transform(input_features)
            
            prediction = rf_model.predict(input_features_scaled)[0]
            probabilities = rf_model.predict_proba(input_features_scaled)[0]
            
            prob_true = probabilities[1]*100
            prob_false = probabilities[0]*100
            
            if prediction == 1:
                st.success(f"Le billet est VRAI avec une probabilité de {prob_true:.2f}%")
            else:
                st.error(f"Le billet est FAUX avec une probabilité de {prob_false:.2f}%")
    
    # ----------------- Section Pièce -----------------
    else:
        st.header("Prédiction de pièces d'euro")
        st.write("Téléversez une image d’une pièce pour obtenir sa valeur prédite.")
        
        uploaded_file = st.file_uploader("Choisis une image de pièce", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image importée", use_container_width=True)
            
            # Télécharger modèle si pas déjà
            file_id = "17r6xrf0vM7wNMrPSaGv8xqUV0cuOl6OE"
            output = "saved_models/model_weights_final.weights.h5"
            if not os.path.exists(output):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
            
            # Mapping classes
            int_to_class = {0:"10c",1:"1c",2:"1e",3:"20c",4:"2c",5:"2e",6:"50c",7:"5c"}
            
            # Créer modèle
            conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
            conv_base.trainable = False
            for layer in conv_base.layers:
                if "block5" in layer.name:
                    layer.trainable = True
            best_model = models.Sequential([
                conv_base,
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.318),
                layers.Dense(8, activation='softmax')
            ])
            best_model.load_weights(output)
            
            # Préparer l'image
            img_resized = image.resize((150,150))
            img_array = img_to_array(img_resized)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            start_time = time.time()
            preds = best_model.predict(img_array)
            end_time = time.time()
            
            predicted_classes = np.argmax(preds, axis=1)
            confidence = np.max(preds)*100
            elapsed_time = end_time - start_time
            
            st.subheader(f"Valeur prédite : **{int_to_class[predicted_classes[0]]}**")
            st.metric(label="Confiance du modèle", value=f"{confidence:.2f} %")
            st.metric(label="Temps de prédiction", value=f"{elapsed_time:.3f} sec")
            
            st.write("### Probabilités pour chaque classe :")
            for i, cls in int_to_class.items():
                st.write(f"{cls} : {preds[0][i]*100:.2f} %")
            
            st.bar_chart({int_to_class[i]: float(preds[0][i]*100) for i in range(len(int_to_class))})
