"""Chargement et decoupage des donnees avec tests/validations."""
from __future__ import annotations

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_STATE, TARGET

logger = logging.getLogger(__name__)

def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Charge les donnees CSV avec une validation de la coherence."""
    if not path.exists():
        raise FileNotFoundError(f"Le fichier de donnees est introuvable : {path}")

    logger.info(f"Chargement des donnees depuis {path}")
    df = pd.read_csv(path)
    
    # Test 1 : Verifier que la colonne cible est presente
    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET}' n'existe pas dans le dataset.")

    # Test 2 : Supprimer les lignes ou la cible est vide ou nulle
    missing_target = df[TARGET].isna().sum()
    if missing_target > 0:
        logger.warning(f"Suppression de {missing_target} lignes sans valeur pour la cible '{TARGET}'.")
        df = df.dropna(subset=[TARGET])
        
    # Test 3 : Verifier que le dataframe n'est pas vide apres nettoyage
    if df.empty:
        raise ValueError("Le dataset est vide apres la suppression des valeurs cible manquantes.")

    # Test 4 : S'assurer du bon type numerique pour la variable de regression
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    invalid_targets = df[TARGET].isna().sum()
    if invalid_targets > 0:
        logger.warning(f"Suppression de {invalid_targets} lignes dont la cible n'etait pas un nombre valide.")
        df = df.dropna(subset=[TARGET])
        
    return df


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Separe le dataset en features (X) et label (y).
    
    Pour la regression, nous retirons la stratification (stratify=y).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # Plus de stratify=y car la cible est continue (regression)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)