"""DAG Airflow - trafic de previsions quotidien pour la regression.

Planifie l'envoi quotidien d'un lot de previsions a l'API : chaque jour a
10h, on echantillonne 20 lignes du jeu de donnees et on les envoie en
POST /predict.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Nombre de previsions envoyees a chaque execution.
N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """Echantillonner N_PREDICTIONS lignes et les envoyer a l'API /predict."""
    import httpx
    from src.data import load_data

    # L'API tourne par defaut sur http://api:8000 en docker
    api_url = os.environ.get("API_URL", "http://api:8000")

    df = load_data()

    # Colonnes attendues par le schema Pydantic de l'API (src/api.py -> Features)
    API_COLUMNS = [
        "Year", "Mileage_kmpl", "Engine_CC", "Horsepower", "Kms_Driven",
        "Insurance_Valid", "Service_History", "Accidents", "Tax_Paid",
        "Number_of_Doors", "Seats", "Registration_Age",
        "Brand", "Model", "Fuel_Type", "Transmission", "Owner_Type",
        "Color", "City",
    ]

    # On ne garde que les colonnes attendues par l'API et on supprime les lignes incompletes
    features = df[API_COLUMNS].dropna()

    # Echantillonner N_PREDICTIONS lignes
    sample = features.sample(n=N_PREDICTIONS)

    # Ouvrir un client httpx sur l'API_URL, verifier /health,
    # puis pour chaque ligne envoyer POST /predict avec le payload JSON.
    with httpx.Client(base_url=api_url, timeout=10.0) as client:
        # Verifier la sante de l'API
        health_resp = client.get("/health")
        health_resp.raise_for_status()
        
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            logger.info("Prediction reussie: %s", response.json())

    logger.info("%d previsions envoyees a %s", N_PREDICTIONS, api_url)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="10 * * * *",
    catchup=False,
    tags=["regression", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
