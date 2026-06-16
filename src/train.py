"""Entrainement du modele de regression (baseline).

Seance 5 - TP MLflow Tracking
    Ce script entraine et evalue un modele SANS aucun suivi d'experience.
    Votre mission : instrumenter cet entrainement avec MLflow (voir les TODO).
"""
from __future__ import annotations

import argparse
import logging
import math

import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR
from src.data import load_data, split
from src.features import build_preprocessor, create_features
from src.tracking import (
    setup_experiment, log_dataset, start_run, log_params, log_metrics, 
    log_model, log_scatter_plot
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_model(alpha: float = 1.0) -> Pipeline:
    """Construit un pipeline de regression avec penalite L2 (Ridge)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def train(alpha: float = 1.0) -> dict:
    logger.info("Chargement des donnees...")
    df = load_data()
    
    logger.info("Creation des variables metier (feature engineering)...")
    df = create_features(df)
    
    logger.info("Decoupage en ensembles d'entrainement et de test...")
    x_train, x_test, y_train, y_test = split(df)

    logger.info("Configuration MLflow via tracking.py...")
    setup_experiment()

    run_name = f"Ridge_alpha={alpha}"
    with start_run(run_name=run_name):
        log_dataset(df, context="training")
        logger.info(f"Debut de l'entrainement du modele (Ridge alpha={alpha})...")
        model = build_model(alpha=alpha)
        model.fit(x_train, y_train)

        logger.info("Entrainement termine. Prediction sur l'ensemble de test...")
        preds = model.predict(x_test)
        
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }
        logger.info(f"Resultats : RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | R2={metrics['r2']:.3f}")

        logger.info("Envoi des metadonnees et du modele a MLflow...")
        log_params({"alpha": alpha})
        log_metrics(metrics)
        log_model(model, "model")
        
        # Creation et log du graphique
        log_scatter_plot(y_test, preds, title="Vrai Prix vs Prix Predit")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")
        
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    train(alpha=args.alpha)


if __name__ == "__main__":
    main()
