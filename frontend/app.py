"""Frontend Streamlit pour tester l'API de prédiction des prix de voitures."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Prédiction Prix Voiture", layout="wide")
st.title("🚗 Estimateur de Prix de Voitures d'Occasion")

api_url = st.sidebar.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prédiction", "Historique"])

with predict_tab:
    st.subheader("Entrez les caractéristiques de la voiture")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Informations Générales")
            brand = st.text_input("Marque (ex: Toyota)", value="Toyota")
            model = st.text_input("Modèle (ex: Corolla)", value="Corolla")
            year = st.number_input("Année de fabrication", min_value=1990, max_value=2025, value=2015, step=1)
            registration_age = st.number_input("Âge d'immatriculation (années)", min_value=0, max_value=40, value=5, step=1)
            color = st.text_input("Couleur (ex: White)", value="White")
            city = st.text_input("Ville (ex: Mumbai)", value="Mumbai")
            
        with col2:
            st.markdown("### Mécanique & Performances")
            engine_cc = st.number_input("Cylindrée (Engine CC)", min_value=500.0, max_value=8000.0, value=1200.0)
            horsepower = st.number_input("Puissance (Horsepower)", min_value=30.0, max_value=1000.0, value=85.0)
            mileage_kmpl = st.number_input("Consommation (Mileage kmpl)", min_value=1.0, max_value=50.0, value=15.5)
            kms_driven = st.number_input("Kilométrage parcouru", min_value=0.0, max_value=1000000.0, value=60000.0)
            fuel_type = st.selectbox("Type de carburant", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
        with col3:
            st.markdown("### Administratif & Configuration")
            seats = st.number_input("Nombre de sièges", min_value=2, max_value=10, value=5, step=1)
            number_of_doors = st.number_input("Nombre de portes", min_value=2, max_value=5, value=5, step=1)
            owner_type = st.selectbox("Type de propriétaire", ["First", "Second", "Third", "Fourth & Above"])
            
            insurance_valid = st.checkbox("Assurance valide", value=True)
            service_history = st.checkbox("Historique de service complet", value=True)
            tax_paid = st.checkbox("Taxe payée", value=True)
            accidents = st.number_input("Nombre d'accidents", min_value=0, max_value=10, value=0, step=1)

        st.markdown("---")
        submitted = st.form_submit_button("💰 Estimer le prix", use_container_width=True)

    if submitted:
        payload = {
            "Year": year,
            "Mileage_kmpl": mileage_kmpl,
            "Engine_CC": engine_cc,
            "Horsepower": horsepower,
            "Kms_Driven": kms_driven,
            "Insurance_Valid": 1 if insurance_valid else 0,
            "Service_History": 1 if service_history else 0,
            "Accidents": accidents,
            "Tax_Paid": 1 if tax_paid else 0,
            "Number_of_Doors": number_of_doors,
            "Seats": seats,
            "Registration_Age": registration_age,
            "Brand": brand,
            "Model": model,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Type": owner_type,
            "Color": color,
            "City": city
        }
        
        with st.spinner("Demande à l'API en cours..."):
            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
                
                st.success("Prédiction réussie !")
                st.metric(label="Prix Estimé", value=f"{result['price']:,.2f} ₹")
                
            except httpx.HTTPError as exc:
                st.error(f"Appel à l'API impossible : {exc}")
            except KeyError:
                st.error(f"Réponse inattendue de l'API : {result}")

with history_tab:
    st.subheader("Historique des prévisions")
    st.info("Aucun journal de prévisions : ajoutez un endpoint /predictions à l'API (bonus).")
    _ = pd
