# src/tuning.py
import xgboost as xgb
import mlflow
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from data_preprocessing import load_and_preprocess
import os
import numpy as np


def tune_hyperparameters():
    """
    Effectue une GridSearchCV pour trouver les meilleurs hyperparamètres.
    """
    # Charger les données
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    # Définir la grille de paramètres
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # Modèle de base
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",  # On veut maximiser l'AUC
        n_jobs=-1,
        verbose=2,
    )

    # Lancer la recherche
    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        grid_search.fit(X_train, y_train)

        # Loguer les meilleurs paramètres
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Entraîner le meilleur modèle sur tout le train
        best_model = grid_search.best_estimator_

        # Évaluer sur le test set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score

        test_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("test_auc", test_auc)

        # Sauvegarder le meilleur modèle
        mlflow.xgboost.log_model(best_model, "best_model")

        print(f"Meilleurs paramètres: {grid_search.best_params_}")
        print(f"Meilleur score CV (AUC): {grid_search.best_score_:.4f}")
        print(f"Score sur le test (AUC): {test_auc:.4f}")

    return best_model


if __name__ == "__main__":
    tune_hyperparameters()
