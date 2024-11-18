# Projet_DataScience_Bernardo_Fabrice.B_Filip

# Projet de Data Science : Prédiction de la qualité du vin et du diabète

## Description

Ce projet a pour objectif de développer une application de Machine Learning pour prédire la qualité du vin et le diabète. Il met en œuvre différentes techniques de Machine Learning, notamment la régression linéaire, les arbres de décision, les forêts aléatoires et les réseaux de neurones (perceptron multicouche).

L'application est développée avec Streamlit pour une interface utilisateur interactive et conviviale.

## Fonctionnalités

L'application propose les fonctionnalités suivantes :

* **Prédiction de la qualité du vin :**
    * Choix du modèle de régression (linéaire, arbre de décision, forêt aléatoire).
    * Sélection des variables explicatives.
    * Ajustement des hyperparamètres du modèle.
    * Entraînement du modèle et évaluation des performances (MSE, R²).
    * Visualisation des prédictions avec des graphiques interactifs.
* **Prédiction du diabète :**
    * Choix du modèle de classification (perceptron multicouche, forêt aléatoire, régression logistique).
    * Prétraitement des données.
    * Entraînement du modèle et évaluation des performances.
    * Affichage des prédictions et de l'historique d'entraînement.
* **Analyse d'images d'ongles :** (à venir)

## Installation

1. **Cloner le dépôt :**  `git clone <url_du_depot>`
2. **Créer un environnement virtuel :** `python3 -m venv .venv`
3. **Activer l'environnement virtuel :** 
    * Linux/macOS : `source .venv/bin/activate`
    * Windows : `.venv\Scripts\activate`
4. **Installer les dépendances :** `pip install -r requirements.txt`

## Exécution

1. **Activer l'environnement virtuel.**
2. **Lancer l'application Streamlit :** `streamlit run scripts/app.py`

## Structure du projet

Utilisez ce code avec précaution.

├── data
│   ├── diabete.csv
│   ├── external.py
│   ├── processed.py
│   └── vin.csv
├── models
│   ├── regression_lineaire
│   │   └── modele_vin.pkl
│   └── reseau_neurones
│       └── modele_diabete.h5
├── notebooks
│   ├── diabete
│   │   └── python
│   │       ├── diabetePerceptron.ipynb
│   │       ├── diabetePerceptron.py
│   │       ├── diabeteRandomForestModel.ipynb
│   │       ├── init.py
│   │       ├── diabetePerceptron.py
│   │       └── diabeteRandomForest.py
│   └── exploration_donnees_vin.ipynb
├── scripts
│   ├── classification.py
│   ├── nails.py
│   └── regression.py
├── src
│   ├── models.py
│   ├── utils.py
│   └── visualization.py
└── streamlit
└── app
├── app.py
├── models
│   ├── init.py
│   ├── diabeted_classes.py
│   └── diabete_prediction_ml_module.py
└── utils
└── init.py


## Données

* **`data/vin.csv` :**  Contient les données sur le vin, avec des caractéristiques physico-chimiques et une note de qualité.
* **`data/diabete.csv` :**  Contient les données sur le diabète, avec des caractéristiques des patients et une variable cible indiquant la présence ou l'absence de diabète.

## Modèles

* **Régression linéaire**
* **Arbre de décision**
* **Forêt aléatoire**
* **Perceptron multicouche**

## Technologies utilisées

* **Python**
* **Pandas**
* **Scikit-learn**
* **TensorFlow**
* **Streamlit**
* **Plotly**

## Auteurs

* Bernardo Estacio Abreu
* Fabrice Bellin
* Filip Dabrowsky

## Licence

[Spécifiez la licence de votre projet ici, par exemple MIT License]

## Remerciements