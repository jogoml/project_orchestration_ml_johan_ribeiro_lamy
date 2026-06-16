"""Configuration centrale du projet de régression.

Ce fichier gère la configuration des variables, les chemins, 
et les paramètres MLFlow pour le projet d'orchestration.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "used_car_price_prediction_1M.csv"
MODEL_DIR = ROOT / "models"

TARGET = "Price"

NUMERIC_FEATURES = [
    "Year", 
    "Mileage_kmpl", 
    "Engine_CC", 
    "Horsepower", 
    "Kms_Driven", 
    "Insurance_Valid", 
    "Service_History", 
    "Accidents", 
    "Tax_Paid", 
    "Number_of_Doors", 
    "Seats", 
    "Registration_Age",
    "Kms_per_Year",
    "Engine_Efficiency"
]

CATEGORICAL_FEATURES = [
    "Brand", 
    "Model", 
    "Fuel_Type", 
    "Transmission", 
    "Owner_Type", 
    "Color", 
    "City",
    "Age_Category"
]

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "used-cars-regression")
MODEL_NAME = os.getenv("MODEL_NAME", "price_predictor")

# Seuils pour la Quality Gate (evaluate.py)
EVAL_RMSE_MAX = float(os.getenv("EVAL_RMSE_MAX", "750000.0"))
EVAL_R2_MIN = float(os.getenv("EVAL_R2_MIN", "0.70"))
