import pandas as pd
from sklearn.compose import ColumnTransformer
from src.features import create_features, build_preprocessor

def test_create_features():
    df = pd.DataFrame({
        "Registration_Age": [0, 5, 20, 35],
        "Kms_Driven": [10000, 50000, 200000, 300000],
        "Horsepower": [100, 150, 300, 500],
        "Engine_CC": [1000, 1500, 3000, 5000]
    })
    
    X_new = create_features(df)
    
    # Kms_per_Year: Registration_Age=0 remplace par 1 -> 10000 / 1 = 10000
    assert X_new.loc[0, "Kms_per_Year"] == 10000.0
    assert X_new.loc[1, "Kms_per_Year"] == 10000.0 # 50000 / 5
    
    # Engine_Efficiency: 100 / 1000 = 0.1
    assert X_new.loc[0, "Engine_Efficiency"] == 0.1
    
    # Age_Category: 0 -> Recent, 5 -> Moyen, 20 -> Vieux, 35 -> Collection
    assert X_new.loc[0, "Age_Category"] == "Recent"
    assert X_new.loc[1, "Age_Category"] == "Moyen"
    assert X_new.loc[2, "Age_Category"] == "Vieux"
    assert X_new.loc[3, "Age_Category"] == "Collection"

def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Vérifie que les transformers sont bien définis (num et cat)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names
