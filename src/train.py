"""Entrainement du modele de regression (baseline).

Seance 5 - TP MLflow Tracking
    Ce script entraine et evalue un modele SANS aucun suivi d'experience.
    Votre mission : instrumenter cet entrainement avec MLflow (voir les TODO).
"""
from __future__ import annotations

import argparse
import math

import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR
from src.data import load_data, split
from src.features import build_preprocessor, create_features


def build_model(alpha: float = 1.0) -> Pipeline:
    """Construit un pipeline de regression avec penalite L2 (Ridge)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def train(alpha: float = 1.0) -> dict:
    df = load_data()
    df = create_features(df)
    
    x_train, x_test, y_train, y_test = split(df)

    model = build_model(alpha=alpha)
    model.fit(x_train, y_train)

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
    print(f"RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}")

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
