"""Script contenant la logique métier de prédiction."""
from __future__ import annotations

import pandas as pd
from src.features import create_features

def predict_price(model, features_dict: dict) -> float:
    """Effectue une prédiction de prix à partir d'un dictionnaire de features brutes.
    
    Parameters
    ----------
    model : 
        Modèle de machine learning chargé (par ex. un pipeline scikit-learn).
    features_dict : dict
        Dictionnaire contenant les caractéristiques brutes de la voiture.
        
    Returns
    -------
    float
        Le prix estimé de la voiture.
    """
    # 1. Conversion en DataFrame (une seule ligne)
    df_raw = pd.DataFrame([features_dict])
    
    # 2. Ajout des variables métier (feature engineering)
    df_features = create_features(df_raw)
    
    # 3. Prédiction
    pred = float(model.predict(df_features)[0])
    
    return pred
