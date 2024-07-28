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
    # Code pour le prétraitement et la formation du modèle si le fichier n'est pas téléchargé
    st.write("En attente du téléchargement du fichier CSV...")
    # Charger les données
    data_path = "C:\\Users\\dell\\Downloads\\Expresso_churn_dataset.csv"
    data = pd.read_csv(data_path)

    # Afficher des informations générales sur l'ensemble de données
    st.write("Aperçu des données :")
    st.write(data.head())
    st.write("Statistiques descriptives :")
    st.write(data.describe())
    st.write("Informations sur les données :")
    st.write(data.info())

    # Gérer les valeurs manquantes
    data = data.dropna()
    st.write("Données après suppression des valeurs manquantes :")
    st.write(data.info())

    # Supprimer les doublons
    data = data.drop_duplicates()
    st.write("Données après suppression des doublons :")
    st.write(data.info())

    # Gérer les valeurs aberrantes (exemple simple)
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    st.write("Données après gestion des valeurs aberrantes :")
    st.write(data.info())

    # Encoder les caractéristiques catégorielles
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    st.write("Données après encodage des caractéristiques catégorielles :")
    st.write(data.info())

    # Afficher les colonnes de l'ensemble de données
    st.write("Colonnes de l'ensemble de données :")
    st.write(data.columns)

    # Séparer les caractéristiques et la cible
    if 'CHURN' in data.columns:  # Correction ici, 'CHURN' au lieu de 'churn'
        X = data.drop('CHURN', axis=1)  # Utilisez la colonne 'CHURN' comme cible
        y = data['CHURN']
    else:
        st.error("La colonne 'CHURN' n'est pas présente dans l'ensemble de données.")
        st.stop()

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Former le modèle
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Précision du modèle : {accuracy * 100:.2f}%")

    # Sauvegarder le modèle et les label encoders
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders, file)

