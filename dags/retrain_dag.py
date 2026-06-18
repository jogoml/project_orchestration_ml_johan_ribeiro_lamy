"""DAG Airflow - pipeline de re-entrainement du modele pour la regression.

Pipeline simple : verification des donnees -> entrainement avec optuna -> controle
qualite.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    """Verification simple de la presence du fichier de donnees."""
    from src.config import DATA_PATH
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Le fichier de donnees est introuvable : {DATA_PATH}")
    
    logger.info("Fichier de donnees trouve : %s", DATA_PATH)


def task_train(**context) -> None:
    """Reentrainement du modele avec Optuna."""
    from src.train_optuna import train
    
    # Entrainement avec Optuna (xgboost, 5 essais, CV=2 pour aller un peu plus vite en reentrainement regulier)
    metrics = train(model_name="xgboost", n_trials=5, cv=2)
    
    # Pousser le r2 dans XCom pour l'etape de validation
    context["ti"].xcom_push(key="r2", value=metrics["r2"])
    logger.info("Entrainement termine. R2=%.3f", metrics["r2"])


def task_check_quality(**context) -> None:
    """Controle qualite du modele reentraine."""
    from src.config import EVAL_R2_MIN
    
    # Recuperer r2 = context["ti"].xcom_pull(task_ids="train", key="r2")
    r2 = context["ti"].xcom_pull(task_ids="train", key="r2")
    
    if r2 is None:
        raise ValueError("Impossible de recuperer le score R2 depuis XCom.")
        
    logger.info("Score R2 recupere : %.3f (Seuil : %.3f)", r2, EVAL_R2_MIN)
    
    # Si r2 < EVAL_R2_MIN, lever une ValueError (le pipeline echoue)
    if r2 < EVAL_R2_MIN:
        raise ValueError(f"Le modele n'atteint pas le seuil de qualite: {r2:.3f} < {EVAL_R2_MIN:.3f}")
    
    # Sinon, logger un message de succes
    logger.info("Validation reussie ! Le modele a passe la porte qualite.")


with DAG(
    dag_id="model_retraining",
    description="Verifie les donnees, reentraine le modele avec Optuna et controle sa qualite",
    schedule="10 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["regression", "training", "optuna"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    # Ordre d'execution
    prepare >> train_task >> check
