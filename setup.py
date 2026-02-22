from setuptools import setup, find_packages

setup(
    name="telco-churn-predictor",
    version="1.0.0",
    author="Chibeni Amine",
    description="Prédiction de désabonnement client avec XGBoost",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "mlflow",
    ],
    python_requires=">=3.12",
)
