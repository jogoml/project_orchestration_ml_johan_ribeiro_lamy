"""Entrainement de modeles avances pour la prediction du prix des voitures.

Ce script permet de choisir dynamiquement le modele a entrainer via argparse.
"""
from __future__ import annotations

import argparse
import logging
import math
import os

import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from src.data import load_data, split
from src.features import build_preprocessor, create_features

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_model(model_name: str) -> Pipeline:
    """Construit un pipeline avec le modele specifie."""
    if model_name == "xgboost":
        regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=2)
    elif model_name == "lightgbm":
        regressor = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=1)
    elif model_name == "mlp":
        regressor = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=50, random_state=42, verbose=True)
    else:
        raise ValueError(f"Modele non supporte : {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )

def train(model_name: str) -> dict:
    logger.info("Chargement des donnees...")
    df = load_data()
    
    logger.info("Creation des variables metier (feature engineering)...")
    df = create_features(df)
    
    logger.info("Decoupage en ensembles d'entrainement et de test...")
    x_train, x_test, y_train, y_test = split(df)

    logger.info(f"Configuration MLflow : URI={MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    pipeline = build_model(model_name)
    regressor = pipeline.named_steps["regressor"]
    
    if model_name in ["xgboost", "lightgbm"]:
        run_name = f"{model_name.upper()}_n={regressor.n_estimators}_lr={regressor.learning_rate}"
    elif model_name == "mlp":
        run_name = f"MLP_layers={regressor.hidden_layer_sizes}_iter={regressor.max_iter}"
    else:
        run_name = model_name

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Debut de l'entrainement du modele ({model_name})...")
        
        # Le pipeline encapsule le preprocess et le regresseur
        pipeline.fit(x_train, y_train)

        logger.info("Entrainement termine. Prediction sur l'ensemble de test...")
        preds = pipeline.predict(x_test)
        
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
        mlflow.log_param("model_name", model_name)
        
        # Recuperation du regresseur dans le pipeline pour logguer ses parametres
        regressor = pipeline.named_steps["regressor"]
        mlflow.log_params(regressor.get_params())
        
        mlflow.log_metrics(metrics)
        # On loggue le pipeline entier (qui inclut le preprocessing)
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Creation et log du graphique
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Vrai Prix")
        ax.set_ylabel("Prix Predit")
        ax.set_title(f"Vrai Prix vs Prix Predit ({model_name})")
        plt.tight_layout()
        fig.savefig("scatter.png")
        mlflow.log_artifact("scatter.png")
        plt.close(fig)
        os.remove("scatter.png")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_DIR / f"model_{model_name}.joblib")
        
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["xgboost", "lightgbm", "mlp"], 
        required=True, 
        help="Choix du modele a entrainer (xgboost, lightgbm ou mlp)"
    )
    args = parser.parse_args()
    train(model_name=args.model)

if __name__ == "__main__":
    main()
