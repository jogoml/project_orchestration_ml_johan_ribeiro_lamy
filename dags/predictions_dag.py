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

    from src.config import TARGET
    from src.data import load_data

    # L'API tourne par defaut sur http://api:8000 en docker
    api_url = os.environ.get("API_URL", "http://api:8000")

    # On retire la colonne cible : l'API ne recoit que les features.
    df = load_data()
    if TARGET in df.columns:
        features = df.drop(columns=[TARGET])
    else:
        features = df

    # Echantillonner N_PREDICTIONS lignes
    sample = features.sample(n=N_PREDICTIONS, random_state=42)

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
    schedule="0 10 * * *",
    catchup=False,
    tags=["regression", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
