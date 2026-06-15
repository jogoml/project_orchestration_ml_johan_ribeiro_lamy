import pandas as pd
import pytest
from pathlib import Path
from src.data import load_data, split
from src.config import TARGET

def test_load_data_success(tmp_path):
    # Créer un fichier CSV temporaire valide
    df = pd.DataFrame({
        "Feature1": [1, 2, 3],
        TARGET: [100, 200, 300]
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    
    loaded_df = load_data(csv_path)
    assert len(loaded_df) == 3
    assert TARGET in loaded_df.columns

def test_load_data_missing_target(tmp_path):
    # Fichier CSV sans la colonne cible
    df = pd.DataFrame({
        "Feature1": [1, 2, 3]
    })
    csv_path = tmp_path / "data_no_target.csv"
    df.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="La colonne cible.*n'existe pas"):
        load_data(csv_path)

def test_load_data_empty_after_drop(tmp_path):
    # Fichier CSV avec uniquement des cibles nulles
    df = pd.DataFrame({
        "Feature1": [1, 2],
        TARGET: [None, pd.NA]
    })
    csv_path = tmp_path / "data_empty.csv"
    df.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="Le dataset est vide apres la suppression"):
        load_data(csv_path)

def test_split():
    df = pd.DataFrame({
        "Feature1": range(100),
        "Feature2": range(100),
        TARGET: range(100)
    })
    
    X_train, X_test, y_train, y_test = split(df, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert TARGET not in X_train.columns
