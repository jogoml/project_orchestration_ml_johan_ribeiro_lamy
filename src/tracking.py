"""Configuration partagee du suivi MLflow."""
from __future__ import annotations

import logging

import os
from typing import Any

import mlflow
import mlflow.data
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant (tracabilite donnees -> modele)."""
    dataset = mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name=name)  # type: ignore
    mlflow.log_input(dataset, context=context)


def start_run(run_name: str):
    """Demarrer un run MLflow."""
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    """Logger un dictionnaire de parametres."""
    mlflow.log_params(params)


def log_param(key: str, value: Any) -> None:
    """Logger un seul parametre."""
    mlflow.log_param(key, value)


def log_metrics(metrics: dict) -> None:
    """Logger des metriques."""
    mlflow.log_metrics(metrics)


def log_model(model, artifact_path: str = "model", registered_model_name: str | None = None) -> None:
    """Logger un modele scikit-learn et l'enregistrer dans le Model Registry."""
    # skops_trusted_types : declarer explicitement les types autorises (XGBoost, LightGBM, numpy)
    # pour satisfaire la validation de securite de MLflow >= 2.12
    trusted_types = [
        "numpy.dtype",
        "numpy.ndarray",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBRegressor",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMRegressor",
    ]
    mlflow.sklearn.log_model(
        model,
        artifact_path,
        registered_model_name=registered_model_name,
        skops_trusted_types=trusted_types,
    )


def log_scatter_plot(y_true, y_pred, title: str = "Vrai Prix vs Prix Predit", filename: str = "scatter.png") -> None:
    """Generer et logger un scatter plot des predictions vs reel."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel("Vrai Prix")
    ax.set_ylabel("Prix Predit")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close(fig)
    if os.path.exists(filename):
        os.remove(filename)
