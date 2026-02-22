# src/predict.py (version qui prend le dernier modèle)
import pandas as pd
import numpy as np
import os
import joblib
import glob
from data_preprocessing import load_and_preprocess


def get_latest_model():
    """Récupère le fichier de modèle le plus récent dans models/"""
    model_files = glob.glob("models/xgboost_model_*.pkl")
    if not model_files:
        return None

    # Trier par date de modification (le plus récent d'abord)
    latest = max(model_files, key=os.path.getmtime)
    return latest


def main():
    print("🚀 PRÉDICTION DE CHURN TELECOM")
    print("=" * 60)

    # 1. Charger les données
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"\n📂 Chargement des données...")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    print(f"   ✅ Test set: {X_test.shape} avec {X_test.shape[1]} features")

    # 2. Trouver le dernier modèle
    model_path = get_latest_model()
    if model_path is None:
        print("\n❌ Aucun modèle trouvé dans models/")
        print("💡 Lance d'abord: python src/train.py")
        return

    print(f"\n🤖 Chargement du modèle: {model_path}")
    model = joblib.load(model_path)
    print(f"   ✅ Modèle chargé")

    # 3. Vérifier la compatibilité des features
    if hasattr(model, "n_features_in_"):
        print(f"   📊 Modèle entraîné avec {model.n_features_in_} features")
        print(f"   📊 Données actuelles avec {X_test.shape[1]} features")

        if model.n_features_in_ != X_test.shape[1]:
            print(f"\n❌ ERREUR: Mismatch de features!")
            print(f"   Modèle attend {model.n_features_in_} features")
            print(f"   Données ont {X_test.shape[1]} features")
            return

    # 4. Prendre 5 clients aléatoires
    np.random.seed(42)
    indices = np.random.choice(len(X_test), 5, replace=False)
    X_sample = X_test.iloc[indices]
    y_true = y_test.iloc[indices]

    # 5. Prédire
    print(f"\n🔮 Prédictions pour 5 clients...")
    y_pred = model.predict(X_sample)
    y_proba = model.predict_proba(X_sample)[:, 1]

    # 6. Afficher les résultats
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DES PRÉDICTIONS")
    print("=" * 60)

    results = []
    for idx, true, pred, proba in zip(indices, y_true, y_pred, y_proba):
        # Déterminer le risque
        if proba > 0.7:
            risk = "🔴 ÉLEVÉ"
        elif proba > 0.3:
            risk = "🟠 MOYEN"
        else:
            risk = "🟢 FAIBLE"

        results.append(
            {
                "Client": f"#{idx}",
                "Réalité": "🚫 PARTI" if true == 1 else "✅ RESTÉ",
                "Prédiction": "🚫 PARTIRA" if pred == 1 else "✅ RESTERA",
                "Proba": f"{proba:.1%}",
                "Risque": risk,
                "Correct": "✅" if pred == true else "❌",
            }
        )

    # Afficher sous forme de tableau
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Statistiques
    accuracy = (y_pred == y_true).mean()
    print(f"\n📈 Accuracy: {accuracy:.0%} ({int(accuracy*5)}/5)")

    # Recommandations business
    at_risk = sum(y_proba > 0.5)
    if at_risk > 0:
        print(f"\n🔴 {at_risk} client(s) à risque identifié(s):")
        for idx, proba in zip(indices, y_proba):
            if proba > 0.5:
                if proba > 0.8:
                    action = "Offrir une réduction immédiate"
                else:
                    action = "Envoyer un email de fidélisation"
                print(f"   • Client {idx}: {action} (risque: {proba:.1%})")


if __name__ == "__main__":
    main()
