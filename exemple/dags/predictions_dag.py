"""DAG Airflow - trafic de previsions quotidien (squelette).

Seance 17 - TP Airflow (suite)
    Planifie l'envoi quotidien d'un lot de previsions a l'API : chaque jour a
    10h, on echantillonne 20 lignes du jeu de donnees et on les envoie en
    POST /predict. Cela simule un flux de previsions en production (chaque
    appel est journalise par l'API) et alimente la boucle de feedback.
    Completez les TODO (S17-6, S17-7, S17-8).

Prerequis : l'API doit etre joignable via `API_URL` (defaut `http://api:8000`
en docker). En pratique : lancer la stack qui expose l'API en plus d'Airflow.
"""
from __future__ import annotations

import json
import logging
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

    from mlproject.config import API_URL, TARGET
    from mlproject.data import load_data

    # On retire la colonne cible : l'API ne recoit que les features.
    features = load_data().drop(columns=[TARGET])

    # TODO (S17-6) : echantillonner N_PREDICTIONS lignes
    #   sample = features.sample(n=N_PREDICTIONS)

    # TODO (S17-7) : ouvrir un client httpx sur API_URL, verifier /health,
    #   puis pour chaque ligne envoyer POST /predict avec le payload JSON.
    #   Astuce pour des types JSON natifs (pas de numpy) :
    #       payload = json.loads(row.to_json())
    #   Exemple de squelette :
    #       with httpx.Client(base_url=API_URL, timeout=10.0) as client:
    #           client.get("/health").raise_for_status()
    #           for _, row in sample.iterrows():
    #               payload = json.loads(row.to_json())
    #               response = client.post("/predict", json=payload)
    #               response.raise_for_status()
    raise NotImplementedError

    logger.info("%d previsions envoyees a %s", N_PREDICTIONS, API_URL)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    # TODO (S17-8) : planifier tous les jours a 10h -> schedule="0 10 * * *"
    schedule=None,
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
