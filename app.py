import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Charger le modèle et les label encoders
def load_model_and_encoders():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    
    return model, label_encoders

# Fonction pour faire des prédictions
def predict_churn(data, model, label_encoders):
    for column in label_encoders:
        if column in data.columns:
            data[column] = label_encoders[column].transform(data[column])
    prediction = model.predict(data)
    return prediction

# Interface utilisateur Streamlit
st.title("Prédiction de Désabonnement des Clients Expresso")
st.write("""
Cette application prédit la probabilité de désabonnement des clients Expresso
à partir de leurs données comportementales.
""")

# Ajouter un champ pour télécharger un fichier CSV
uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données téléchargées:")
    st.write(data.head())

    # Afficher les colonnes de l'ensemble de données téléchargé
    st.write("Colonnes du fichier téléchargé:")
    st.write(data.columns)

    # Charger le modèle et les label encoders
    model, label_encoders = load_model_and_encoders()

    # Faire les prédictions
    if st.button("Prédire"):
        with st.spinner("Prédiction en cours..."):
            prediction = predict_churn(data, model, label_encoders)
            st.write("Prédictions de désabonnement (0 = Non, 1 = Oui):")
            st.write(prediction)
else:
    # Exemple de données intégrées si aucun fichier n'est téléchargé
    st.write("Aucun fichier téléchargé. Utilisation d'un exemple de jeu de données intégré.")
    data = pd.DataFrame({
        'REGION': ['Dakar', 'Dakar', 'Thies'],
        'TENURE': [12, 24, 36],
        'MONTANT': [1000, 1500, 2000],
        'FREQUENCE_RECH': [10, 15, 20],
        'REVENUE': [5000, 6000, 7000],
        'ARPU_SEGMENT': [50, 60, 70],
        'FREQUENCE': [5, 10, 15],
        'DATA_VOLUME': [1, 2, 3],
        'ON_NET': [100, 150, 200],
        'ORANGE': [50, 60, 70],
        'TIGO': [20, 30, 40],
        'ZONE1': [1, 0, 1],
        'ZONE2': [0, 1, 0],
        'MRG': [10, 20, 30],
        'REGULARITY': [1, 1, 1],
        'TOP_PACK': ['A', 'B', 'A'],
        'FREQ_TOP_PACK': [3, 4, 5],
    })

    st.write("Aperçu des données intégrées :")
    st.write(data.head())

    # Afficher les colonnes de l'ensemble de données intégré
    st.write("Colonnes de l'ensemble de données intégré :")
    st.write(data.columns)

    # Charger le modèle et les label encoders
    model, label_encoders = load_model_and_encoders()

    # Faire les prédictions sur les données intégrées
    if st.button("Prédire sur les données intégrées"):
        with st.spinner("Prédiction en cours..."):
            prediction = predict_churn(data, model, label_encoders)
            st.write("Prédictions de désabonnement (0 = Non, 1 = Oui):")
            st.write(prediction)
