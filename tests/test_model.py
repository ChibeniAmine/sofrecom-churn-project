# tests/test_model.py
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_preprocess
from src.train import train_model


def test_model_performance():
    """Teste que le modèle atteint des performances minimales"""
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Vérifier que le fichier existe
    assert os.path.exists(data_path), f"Fichier non trouvé: {data_path}"

    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    # Entraînement rapide pour le test
    model, metrics, _ = train_model(
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
    )

    assert metrics["accuracy"] > 0.70, f"Accuracy trop basse: {metrics['accuracy']}"
    assert metrics["roc_auc"] > 0.75, f"AUC trop basse: {metrics['roc_auc']}"

    print("✅ Test de performance réussi!")


def test_model_prediction_shape():
    """Teste que les prédictions ont la bonne forme"""
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    model, _, _ = train_model({"n_estimators": 50, "max_depth": 3})

    predictions = model.predict(X_test.head(10))
    probabilities = model.predict_proba(X_test.head(10))[:, 1]

    assert len(predictions) == 10
    assert (
        all(predictions.isin([0, 1]))
        if hasattr(predictions, "isin")
        else all(p in [0, 1] for p in predictions)
    )
    assert len(probabilities) == 10
    assert all(0 <= p <= 1 for p in probabilities)

    print("✅ Test de prédiction réussi!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
