import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle
model = joblib.load('model.pkl')

# Configuration de l'interface utilisateur
st.title("Prédiction de l'Évolution des Patients")
st.write("Cette application prédit l'évolution des patients à partir de paramètres médicaux.")

# Formulaire pour collecter les entrées utilisateur
age = st.number_input("Âge", min_value=1, value=30)
sexe = st.selectbox("Sexe", options=["Homme", "Femme"])
PremiersSigne = st.number_input("Premiers Signe - Admission à l'hopital", min_value=1, value=30)
AdmissionHopital = st.number_input("Admission à l'hopital - Prise en charge medicale", min_value=1, value=30)
hypertension = st.selectbox("Hypertension Artérielle", options=["OUI", "NON"])
diabete = st.selectbox("Diabète", options=["OUI", "NON"])
cardiopathie = st.selectbox("Cardiopathie", options=["OUI", "NON"])
hemiplegie = st.selectbox("Hémiplégie", options=["OUI", "NON"])
paralysie_facial = st.selectbox("Paralysie Faciale", options=["OUI", "NON"])
aphasie = st.selectbox("Aphasie", options=["OUI", "NON"])
hemiparesie = st.selectbox("Hémiparésie", options=["OUI", "NON"])
engagement_cerebral = st.selectbox("Engagement Cérébral", options=["OUI", "NON"])
inondation_ventriculaire = st.selectbox("Inondation Ventriculaire", options=["OUI", "NON"])
traitement = st.selectbox("Traitement", options=["Thrombolyse", "Chirurgie"])
temps_suivi = st.number_input("Temps de Suivi après Traitement (en jours)", min_value=1, value=100)

# Conversion des entrées utilisateur en format compatible avec le modèle
data = {
    "AGE": age,
    "SEXE": 1 if sexe == "Homme" else 0,
    "Premiers Signe - Admission à l'hopital": PremiersSigne,
    "Admission à l'hopital - Prise en charge medicale": AdmissionHopital,
    "Hypertension Artérielle": 1 if hypertension == "OUI" else 0,
    "Diabète": 1 if diabete == "OUI" else 0,
    "Cardiopathie": 1 if cardiopathie == "OUI" else 0,
    "Hémiplégie": 1 if hemiplegie == "OUI" else 0,
    "Paralysie Faciale": 1 if paralysie_facial == "OUI" else 0,
    "Aphasie": 1 if aphasie == "OUI" else 0,
    "Hémiparésie": 1 if hemiparesie == "OUI" else 0,
    "Engagement Cérébral": 1 if engagement_cerebral == "OUI" else 0,
    "Inondation Ventriculaire": 1 if inondation_ventriculaire == "OUI" else 0,
    "Traitement": 1 if traitement == "Thrombolyse" else 0,
    "Temps de Suivi après Traitement (en jours)": temps_suivi,
}

# Créez un bouton pour effectuer la prédiction
if st.button("Prédire"):
    try:
        # Préparez les données pour le modèle
        input_values = list(data.values())
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)

        # Affichez le résultat
        if prediction[0] == 1:
            st.success("Résultat : Le patient est prédit comme Vivant.")
        else:
            st.error("Résultat : Le patient est prédit comme Décédé.")
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
