"""Entrainement de modeles avec optimisation Optuna pour la regression."""
from __future__ import annotations

import argparse
import logging
import math

import joblib
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import set_config

from src.config import MODEL_DIR, MODEL_NAME
from src.data import load_data, split
from src.features import build_preprocessor, create_features
from src.tracking import (
    setup_experiment, log_dataset, start_run, log_param, log_params, 
    log_metrics, log_model, log_scatter_plot
)

set_config(transform_output="pandas")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_pipeline(model_name: str, params: dict) -> Pipeline:
    """Construit le pipeline avec les hyperparamètres donnés par Optuna."""
    if model_name == "xgboost":
        regressor = XGBRegressor(random_state=42, verbosity=0, **params)
    elif model_name == "lightgbm":
        regressor = LGBMRegressor(random_state=42, verbose=-1, **params)
    elif model_name == "mlp":
        regressor = MLPRegressor(random_state=42, verbose=False, max_iter=50, **params)
    else:
        raise ValueError(f"Modele non supporte : {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
    return pipeline

def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Définit l'espace de recherche Optuna pour chaque modèle."""
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    elif model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    elif model_name == "mlp":
        hidden_layer_size = trial.suggest_categorical("hidden_layer_sizes", [(5,), (10,), (20,)])  # type: ignore
        return {
            "hidden_layer_sizes": hidden_layer_size,
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
        }
    else:
        raise ValueError(f"Modele non supporte : {model_name}")

def objective(trial: optuna.Trial, model_name: str, x_train, y_train, cv: int) -> float:
    """Fonction objectif d'Optuna à maximiser (cross_val_score)."""
    params = suggest_params(trial, model_name)
    pipeline = build_pipeline(model_name, params)
    
    # cross_val_score avec neg_root_mean_squared_error renvoie des valeurs négatives.
    # Optuna maximise par défaut, donc une valeur moins négative (proche de 0) est meilleure.
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=2)
    return scores.mean()

def train(model_name: str, n_trials: int = 10, cv: int = 2) -> dict:
    logger.info("Chargement des donnees...")
    df = load_data()
    
    logger.info("Creation des variables metier (feature engineering)...")
    df = create_features(df)
    
    logger.info("Decoupage en ensembles d'entrainement et de test...")
    x_train, x_test, y_train, y_test = split(df)

    logger.info("Configuration MLflow via tracking.py...")
    setup_experiment()

    run_name = f"{model_name.upper()}_Optuna_Trials={n_trials}_CV={cv}"

    with start_run(run_name=run_name):
        log_dataset(df, context="training")
        logger.info(f"Debut de l'entrainement et Optuna pour ({model_name})...")
        
        # Optuna cherche à maximiser (car neg_root_mean_squared_error est négatif)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, model_name, x_train, y_train, cv), n_trials=n_trials)
        
        logger.info(f"Meilleurs hyperparametres trouves : {study.best_params}")
        
        # Ré-entraîner le meilleur pipeline sur tout le jeu d'entrainement
        best_pipeline = build_pipeline(model_name, study.best_params)
        best_pipeline.fit(x_train, y_train)

        logger.info("Entrainement termine. Prediction sur l'ensemble de test...")
        preds = best_pipeline.predict(x_test)
        
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }
        logger.info(f"Resultats finaux : RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | R2={metrics['r2']:.3f}")

        logger.info("Envoi des metadonnees et du modele a MLflow...")
        log_param("model_name", model_name)
        log_param("cv_folds", cv)
        log_param("n_trials", n_trials)
        log_params(study.best_params)
        log_metrics(metrics)
        log_model(best_pipeline, "model", registered_model_name=MODEL_NAME)
        
        log_scatter_plot(y_test, preds, title=f"Vrai Prix vs Prix Predit ({model_name}) - Optuna")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, MODEL_DIR / f"model_{model_name}_optuna.joblib")
        
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
        help="Nombre de folds pour la validation croisee (defaut: 2)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Nombre d'essais Optuna (defaut: 10)"
    )
    args = parser.parse_args()
    train(model_name=args.model, cv=args.cv, n_trials=args.n_trials)

if __name__ == "__main__":
    main()
