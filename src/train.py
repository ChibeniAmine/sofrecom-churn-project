# src/train.py (version avec sauvegarde locale forcée)
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_preprocessing import load_and_preprocess
import argparse
from datetime import datetime

def train_model(params=None):
    """
    Entraîne un modèle XGBoost avec MLflow tracking et sauvegarde locale
    """
    # 1. Préparer les données
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    
    # 2. Définir les paramètres par défaut
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 100,
            'random_state': 42,
            'use_label_encoder': False
        }
    
    # 3. Démarrer un run MLflow
    with mlflow.start_run():
        # Loguer les paramètres
        mlflow.log_params(params)
        
        # Entraîner le modèle
        print("\n🚀 Entraînement du modèle XGBoost...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculer les métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        # Loguer les métriques
        mlflow.log_metrics(metrics)
        
        # 🔴 SAUVEGARDE LOCALE OBLIGATOIRE
        os.makedirs('models', exist_ok=True)
        
        # Générer un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_model_path = f'models/xgboost_model_{timestamp}.pkl'
        
        # Sauvegarder le modèle
        joblib.dump(model, local_model_path)
        print(f"✅ Modèle sauvegardé LOCALEMENT: {local_model_path}")
        
        # Sauvegarder aussi la liste des features (important!)
        feature_names = list(X_train.columns)
        joblib.dump(feature_names, 'models/feature_names.pkl')
        print(f"✅ Features sauvegardées: {len(feature_names)} features")
        
        # Sauvegarder les features importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        print(f"✅ Feature importances sauvegardées")
        
        # Logger le modèle dans MLflow
        mlflow.xgboost.log_model(model, "model")
        print(f"✅ Modèle loggé dans MLflow (run_id: {mlflow.active_run().info.run_id})")
        
        print(f"\n📊 MÉTRIQUES:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n✅ Entraînement terminé avec succès!")
        print(f"   Modèle local: {local_model_path}")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        
        return model, metrics, local_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=100)
    args = parser.parse_args()
    
    params = {
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators
    }
    
    model, metrics, model_path = train_model(params)
