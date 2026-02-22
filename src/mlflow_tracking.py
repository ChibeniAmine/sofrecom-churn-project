"""
Module de tracking MLflow pour le projet de prédiction de churn.
Ce fichier configure et teste le système de tracking des expériences.
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow_tracking():
    """
    Configure MLflow pour le tracking local des expériences.
    Les données seront stockées dans le dossier mlruns à la racine du projet.
    """

    # Définition du chemin ABSOLU pour être sûr
    project_root = Path(__file__).parent.parent  # Remonte de src/ vers la racine
    mlruns_path = project_root / "mlruns"

    # Configuration du tracking URI en format file://
    tracking_uri = f"file://{mlruns_path.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"✅ Tracking URI configuré: {tracking_uri}")

    # Création ou configuration de l'expérience
    experiment_name = "customer-churn-xgboost"

    try:
        # Vérifie si l'expérience existe déjà
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Crée une nouvelle expérience
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=str(
                    project_root / "models"
                ),  # Optionnel: où sauvegarder les modèles
            )
            logger.info(
                f"✅ Nouvelle expérience créée: {experiment_name} (ID: {experiment_id})"
            )
        else:
            logger.info(f"✅ Expérience existante trouvée: {experiment_name}")

        # Active l'expérience
        mlflow.set_experiment(experiment_name)

    except Exception as e:
        logger.error(f"❌ Erreur lors de la configuration de l'expérience: {e}")
        raise

    return tracking_uri


def test_mlflow_logging():
    """
    Test simple pour vérifier que MLflow fonctionne correctement.
    À exécuter pour valider la configuration.
    """

    with mlflow.start_run(run_name="test-configuration") as run:
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)

        # Log metrics
        mlflow.log_metric("accuracy", 0.81)
        mlflow.log_metric("roc_auc", 0.85)

        # Log un tag pour identifier le test
        mlflow.set_tag("purpose", "test-configuration")
        mlflow.set_tag("timestamp", "2026-02-20")

        run_id = run.info.run_id
        logger.info(f"✅ Test run completed avec succès!")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Pour voir les résultats: mlflow ui --port 5000")

        return run_id


if __name__ == "__main__":
    # Exécution du test quand le fichier est lancé directement
    print("🚀 Configuration de MLflow pour le projet...")
    tracking_uri = setup_mlflow_tracking()
    print(f"📊 Tracking URI: {tracking_uri}")
    print("\n🧪 Test de logging...")
    run_id = test_mlflow_logging()
    print(f"\n✨ Configuration réussie!")
