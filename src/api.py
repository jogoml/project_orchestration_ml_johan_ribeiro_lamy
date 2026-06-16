"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    /health est fourni et fonctionne. A vous d'implementer le schema d'entree
    (adapte a VOTRE jeu de donnees), le schema de sortie, le chargement du
    modele et l'endpoint /predict (voir les TODO S12-n).
    Lancement : `uvicorn mlproject.api:app --reload`
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import joblib
from contextlib import asynccontextmanager
from typing import AsyncIterator

from src.config import MODEL_DIR
from src.evaluate import latest_model_uri
from src.script.predict import predict_price
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        model_uri = latest_model_uri()
        ml["model"] = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Modele charge depuis MLflow: {model_uri}")
    except Exception as e:
        logger.error(f"Erreur de chargement MLflow ({e}), tentative depuis le dossier local...")
        # Fallback pour tenter de charger le dernier fichier local si on n'a pas accès à MLflow
        import glob
        models = glob.glob(str(MODEL_DIR / "*.joblib"))
        if models:
            latest_model = max(models, key=os.path.getctime)
            ml["model"] = joblib.load(latest_model)
            logger.info(f"Modele local charge : {latest_model}")
        else:
            logger.warning("Aucun modele n'a pu etre charge.")
            
    yield
    ml.clear()

app = FastAPI(title="Car Price Prediction API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    Year: int = Field(..., description="Année de fabrication")
    Mileage_kmpl: float = Field(..., description="Consommation")
    Engine_CC: float = Field(..., description="Cylindrée")
    Horsepower: float = Field(..., description="Puissance")
    Kms_Driven: float = Field(..., description="Kilométrage")
    Insurance_Valid: int = Field(..., description="Validité de l'assurance")
    Service_History: int = Field(..., description="Historique de service")
    Accidents: int = Field(..., description="Nombre d'accidents")
    Tax_Paid: int = Field(..., description="Taxe payée")
    Number_of_Doors: int = Field(..., description="Nombre de portes")
    Seats: int = Field(..., description="Nombre de sièges")
    Registration_Age: int = Field(..., description="Âge de la voiture")
    Brand: str = Field(..., description="Marque")
    Model: str = Field(..., description="Modèle")
    Fuel_Type: str = Field(..., description="Type de carburant")
    Transmission: str = Field(..., description="Transmission")
    Owner_Type: str = Field(..., description="Type de propriétaire")
    Color: str = Field(..., description="Couleur")
    City: str = Field(..., description="Ville")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Year": 2015,
                "Mileage_kmpl": 15.5,
                "Engine_CC": 1200.0,
                "Horsepower": 85.0,
                "Kms_Driven": 60000.0,
                "Insurance_Valid": 1,
                "Service_History": 1,
                "Accidents": 0,
                "Tax_Paid": 1,
                "Number_of_Doors": 5,
                "Seats": 5,
                "Registration_Age": 5,
                "Brand": "Toyota",
                "Model": "Corolla",
                "Fuel_Type": "Petrol",
                "Transmission": "Manual",
                "Owner_Type": "First",
                "Color": "White",
                "City": "Mumbai"
            }]
        }
    }


class PredictionOut(BaseModel):
    price: float = Field(..., description="Prix prédit de la voiture")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    
    pred = predict_price(model, features.model_dump())
    
    return PredictionOut(price=round(pred, 2))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
