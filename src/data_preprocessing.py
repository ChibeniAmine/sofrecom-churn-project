# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import os


def load_and_preprocess(filepath):
    """
    Charge le dataset Telco Churn et applique le preprocessing.
    Retourne: X_train, X_test, y_train, y_test, feature_names, preprocessor (si besoin)
    """
    # 1. Chargement
    df = pd.read_csv(filepath)

    # 2. Nettoyage de base
    # Supprimer l'ID client (inutile pour la prédiction)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convertir TotalCharges en numérique, forcer les erreurs en NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Gérer les valeurs manquantes (les NaN dans TotalCharges)
    # On peut les remplir par la valeur 0 ou la médiane. La médiane est plus robuste.
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # 4. Encoder la variable cible 'Churn' (Yes=1, No=0)
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])  # 1 pour 'Yes', 0 pour 'No'

    # 5. Séparer features (X) et target (y)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 6. Identifier les colonnes numériques et catégorielles
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # 7. One-Hot Encoding pour les catégorielles
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    # drop_first=True évite le piège de la multicolinéarité et réduit la dimension

    # 8. Split Train/Test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  # stratify=y pour garder la proportion de Churn dans train et test

    # 9. (Optionnel) Standardisation des features numériques
    # XGBoost n'en a pas besoin, mais ça ne fait pas de mal. On peut le commenter.
    # scaler = StandardScaler()
    # X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    # X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    print(f"Train set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Churn rate in train: {y_train.mean():.2f}")

    return X_train, X_test, y_train, y_test


# Pour tester le script directement
if __name__ == "__main__":
    # Chemin relatif depuis la racine du projet
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    print("Preprocessing terminé avec succès.")
