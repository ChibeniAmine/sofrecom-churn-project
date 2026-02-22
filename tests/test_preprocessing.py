# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_preprocess

def test_load_and_preprocess_returns_dataframes():
    """Teste si la fonction retourne bien 4 éléments de type DataFrame/Series."""
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_target_encoding():
    """Teste si la cible est bien encodée en 0 et 1."""
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}

def test_no_missing_values():
    """Teste s'il n'y a plus de valeurs manquantes."""
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
