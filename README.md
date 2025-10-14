# 💶 Prédicteur de Pièces d’Euro

![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 **Objectif du projet**

Développer une application d’**intelligence artificielle** capable de reconnaître automatiquement la valeur d’une **pièce d’euro** à partir d’une simple photo.  
L’application repose sur un modèle de **Deep Learning (VGG16 + couches denses personnalisées)** intégré dans une interface **Streamlit** intuitive.

---

## ⚙️ **Fonctionnalités principales**

- 📸 **Téléversement d’image** d’une pièce (formats : `.jpg`, `.jpeg`, `.png`)  
- 🧠 **Prédiction automatique** de la valeur de la pièce : `1c`, `2c`, `5c`, `10c`, `20c`, `50c`, `1€`, `2€`
- 📊 **Affichage des probabilités** de chaque classe via un graphique interactif
- ⚡ **Temps d’inférence** et **niveau de confiance** affichés en direct
- 🧩 **Téléchargement automatique des poids du modèle** depuis Google Drive

---

## 🧩 **Architecture du modèle final**

Le modèle final est basé sur **VGG16 (ImageNet)** en *feature extraction* + *fine-tuning* partiel des couches du bloc 5.

**Structure simplifiée :**
```python
VGG16 (include_top=False)
↓
Flatten()
↓
Dense(256, activation='relu')
↓
Dropout(0.32)
↓
Dense(8, activation='softmax')
