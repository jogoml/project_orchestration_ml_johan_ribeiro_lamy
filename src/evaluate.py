"""Evaluation automatisee et validation du modele de regression."""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from src.config import (
    DATA_PATH,
    EVAL_RMSE_MAX,
    EVAL_R2_MIN,
    MODEL_NAME,
    TARGET,
)
from src.data import load_data, split
from src.features import create_features
from src.tracking import setup_experiment, start_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def latest_model_uri() -> str:
    """Resoudre l'URI de la derniere version enregistree de MODEL_NAME."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour '{MODEL_NAME}'. "
            "Lancez d'abord un entrainement avec enregistrement."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"

def build_thresholds() -> dict[str, MetricThreshold]:
    """Construire les seuils de validation a partir de la configuration."""
    return {
        # RMSE : plus petit est meilleur
        "root_mean_squared_error": MetricThreshold(
            threshold=EVAL_RMSE_MAX,
            greater_is_better=False
        ),
        # R2 : plus grand est meilleur
        "r2_score": MetricThreshold(
            threshold=EVAL_R2_MIN,
            greater_is_better=True
        ),
    }

def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluer un modele du registry et valider les seuils."""
    df = load_data()
    df = create_features(df)
    _, x_test, _, y_test = split(df)
    
    # mlflow.evaluate attend un seul DataFrame contenant features + cible.
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    # Configuration du tracking
    setup_experiment()
    
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with start_run(run_name="evaluate"):
        # tracabilite -> logger le jeu d'evaluation comme dataset MLflow
        dataset = mlflow.data.from_pandas(eval_df, source=str(DATA_PATH), targets=TARGET, name="eval")  # type: ignore
        mlflow.log_input(dataset, context="evaluation")
        
        # evaluer le modele avec le type "regressor"
        result = mlflow.models.evaluate(
            model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="regressor",
            evaluators=["default"]
        )
        
        logger.info(
            "r2_score=%.3f rmse=%.3f",
            result.metrics["r2_score"],
            result.metrics["root_mean_squared_error"]
        )
        
        # si validate, appliquer la porte qualite
        if validate:
            mlflow.validate_evaluation_results(build_thresholds(), result)
            logger.info("Validation reussie ! Le modele a passe la porte qualite.")
            
        return result

def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=None,
        help="URI du modele a evaluer (defaut: derniere version de MODEL_NAME)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Evalue sans appliquer la porte qualite (seuils)",
    )
    args = parser.parse_args()

    model_uri = args.model_uri or None
    try:
        evaluate_model(model_uri=model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
