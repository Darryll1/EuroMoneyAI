# Prédiction et Vérification de Monnaie : Pièces d’Euros et Billets Frauduleux

![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  **Objectif du projet**

Développer une application d’**intelligence artificielle** capable de reconnaître automatiquement la valeur d’une **pièce d’euro** à partir d’une simple photo.  
L’application repose sur un modèle de **Deep Learning (VGG16 + couches denses personnalisées)** intégré dans une interface **Streamlit** intuitive.

---

##  **Fonctionnalités principales**

-  **Téléversement d’image** d’une pièce (formats : `.jpg`, `.jpeg`, `.png`)  
-  **Prédiction automatique** de la valeur de la pièce : `1c`, `2c`, `5c`, `10c`, `20c`, `50c`, `1€`, `2€`
-  **Affichage des probabilités** de chaque classe via un graphique interactif
-  **Temps d’inférence** et **niveau de confiance** affichés en direct
-  **Téléchargement automatique des poids du modèle** depuis Google Drive

---

##  **Architecture du modèle final**

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
```

# Application de Reconnaissance de Pièces

Les poids finaux (`model_weights_final.weights.h5`) sont automatiquement téléchargés au lancement de l’application.

---

## Comparaison des modèles développés

Durant la phase de recherche, 5 modèles ont été conçus et comparés pour déterminer la meilleure approche.

| # | Modèle | Type | Data Augmentation | Régularisation | Optimisation | Normalisation | Fine-Tuning | Callbacks |
|---|--------|------|-----------------|----------------|-------------|---------------|-------------|-----------|
| 1 | CNN simple | CNN baseline | ❌ | ❌ | Manuel | ❌ | ❌ | ❌ |
| 2 | CNN optimisé | CNN + Optuna | ✅ | ✅ | Optuna | ✅ | ❌ | ✅ |
| 3 | VGG16 (Feature Extraction) | Pré-entraînement | ❌ | ✅ | Optuna | ✅ | ❌ | ✅ |
| 4 | VGG16 (Feature Extraction + Data Augmentation) | Avancé | ✅ | ✅ | Optuna | ✅ | ❌ | ✅ |
| 5 | VGG16 Fine-Tuning | Final | ✅ | ✅ | Optuna | ✅ | ✅ | ✅ |

**Résultat :**  
Le modèle 5 (VGG16 Fine-Tuning) a obtenu la meilleure précision et robustesse, justifiant son intégration dans l’application Streamlit.

---

## Technologies utilisées

- **Langage :** Python  
- **Framework IA :** TensorFlow / Keras  
- **Optimisation :** Optuna, MLflow  
- **Front-End :** Streamlit  
- **Librairies :** NumPy, Pillow, Matplotlib, Seaborn  
- **Modèle de base :** VGG16 (ImageNet)  

---

