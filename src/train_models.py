"""Entrainement de modeles avances pour la prediction du prix des voitures.

Ce script permet de choisir dynamiquement le modele a entrainer via argparse.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Tuple, Dict, Any

import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import set_config

set_config(transform_output="pandas")

from src.config import MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from src.data import load_data, split
from src.features import build_preprocessor, create_features

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_model(model_name: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """Construit un pipeline et sa grille de parametres pour GridSearchCV."""
    if model_name == "xgboost":
        regressor = XGBRegressor(random_state=42, verbosity=0)
        param_grid = {
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [5, 7], 
            "regressor__subsample": [0.5],
        }
    elif model_name == "lightgbm":
        regressor = LGBMRegressor(random_state=42, verbose=-1, subsample_freq=1)
        param_grid = {
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__num_leaves": [31, 63],
            "regressor__subsample": [0.5],
        }
    elif model_name == "mlp":
        regressor = MLPRegressor(max_iter=50, random_state=42, verbose=False)
        param_grid = {
            "regressor__hidden_layer_sizes": [(5,),(10,)],
            "regressor__learning_rate_init": [0.001, 0.01],
            "regressor__alpha": [0.0001, 0.01],
        }
    else:
        raise ValueError(f"Modele non supporte : {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
    return pipeline, param_grid

def train(model_name: str, cv: int = 2) -> dict:
    logger.info("Chargement des donnees...")
    df = load_data()
    
    logger.info("Creation des variables metier (feature engineering)...")
    df = create_features(df)
    
    logger.info("Decoupage en ensembles d'entrainement et de test...")
    x_train, x_test, y_train, y_test = split(df)

    logger.info(f"Configuration MLflow : URI={MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    pipeline, param_grid = build_model(model_name)
    
    grid = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring="neg_root_mean_squared_error", 
        n_jobs=2,
        verbose=1
    )

    run_name = f"{model_name.upper()}_GridSearch_CV={cv}"

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Debut de l'entrainement et GridSearchCV pour ({model_name})...")
        
        # Le grid search gère l'entraînement
        grid.fit(x_train, y_train)
        
        best_model = grid.best_estimator_

        logger.info("Entrainement termine. Prediction sur l'ensemble de test...")
        preds = best_model.predict(x_test)
        
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
        mlflow.log_param("cv_folds", cv)
        
        # Log des meilleurs parametres trouves
        logger.info(f"Meilleurs parametres : {grid.best_params_}")
        mlflow.log_params(grid.best_params_)
        
        mlflow.log_metrics(metrics)
        # On loggue le meilleur pipeline (qui inclut le preprocessing)
        mlflow.sklearn.log_model(best_model, "model")
        
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
        joblib.dump(best_model, MODEL_DIR / f"model_{model_name}.joblib")
        
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
    parser.add_argument(
        "--cv",
        type=int,
        default=2,
        help="Nombre de folds pour la validation croisee (defaut: 3)"
    )
    args = parser.parse_args()
    train(model_name=args.model, cv=args.cv)

if __name__ == "__main__":
    main()
