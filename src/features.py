"""Construction du pre-processing."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Cree un pipeline de transformation gerant :
    1. Les variables quantitatives : imputation (mediane) et mise a l'echelle (StandardScaler).
    2. Les variables qualitatives : imputation (mode) et encodage (OneHotEncoder).
    """
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

def create_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ajoute de nouvelles variables métiers au dataframe."""
    X = X.copy()
    
    # 1. Kms_per_Year : Kilométrage annuel moyen
    # On remplace 0 par 1 pour éviter la division par zéro (voiture de moins d'un an)
    safe_age = X["Registration_Age"].replace(0, 1)
    X["Kms_per_Year"] = X["Kms_Driven"] / safe_age
    
    # 2. Engine_Efficiency : Ratio Puissance / Cylindrée
    safe_engine = X["Engine_CC"].replace(0, 1)
    X["Engine_Efficiency"] = X["Horsepower"] / safe_engine
    
    # 3. Age_Category : Discrétisation de l'âge de la voiture
    X["Age_Category"] = pd.cut(
        X["Registration_Age"], 
        bins=[-1, 4, 10, 25, 200], 
        labels=["Recent", "Moyen", "Vieux", "Collection"]
    ).astype(str)
    
    return X


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    preprocessor = build_preprocessor()
    print(preprocessor)