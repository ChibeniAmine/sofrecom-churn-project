# 📞 Telco Customer Churn Prediction - MLOps Project

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-green.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Problème métier
Prédire le désabonnement client (churn) pour une entreprise de télécommunications afin de permettre des actions de fidélisation ciblées.

## 📊 Dataset
- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Clients**: 7 043
- **Features**: 20 (démographiques, comptes, services)
- **Target**: Churn (27% de clients désabonnés)

## 🏗️ Architecture du projet
sofrecom-churn-project/
├── .github/               # Workflows CI/CD
├── data/                  # Données brutes (gitignoré)
├── notebooks/             # EDA et expérimentations
├── src/                   # Code source modulaire
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── tuning.py
│   └── predict.py
├── tests/                 # Tests unitaires
│   ├── test_preprocessing.py
│   └── test_model.py      #
├── models/                # Modèles sauvegardés (gitignoré)
├── mlruns/                # Tracking MLflow (gitignoré)
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py               # Pour packaging


## 🚀 Installation

```bash
git clone https://github.com/tonpseudo/sofrecom-churn-project.git
cd sofrecom-churn-project
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows
pip install -r requirements.txt

📥 Données
# Télécharger depuis Kaggle (nécessite API key)
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip

🏃 Utilisation
1. Exploration (EDA)
jupyter notebook notebooks/01_eda.ipynb

2. Entraînement
# Modèle de base
python src/train.py
# Avec paramètres personnalisés
python src/train.py --learning_rate 0.05 --max_depth 7

3. Optimisation
python src!tuning:py

4. Visualisation des runs
mlflow ui

5. Tests
pytest tests/ -v

6. Prédiction
# Utilise automatiquement le meilleur modèle
python src/predict.py
