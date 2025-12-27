# tests/test_model_preprocessing.py
import pandas as pd
import numpy as np
import pytest
from src.model_preprocessing import prepare_data_for_modeling

def test_prepare_data_for_modeling_smote():
    # Small synthetic dataset
    X = pd.DataFrame({
        'num1': np.random.randn(100),
        'cat1': ['A', 'B'] * 50
    })
    y = pd.Series([0]*90 + [1]*10)  # 10% minority
    
    X_train_bal, y_train_bal, X_test, y_test, prep = prepare_data_for_modeling(
        X, y, "Test", "smote", test_size=0.3, random_state=42
    )
    
    # Check shapes
    assert len(X_train_bal) > len(X_train_bal) * 0.9  # Should be roughly balanced
    assert y_train_bal.value_counts(normalize=True).min() > 0.4  # ~50/50

def test_prepare_data_for_modeling_invalid_input():
    X = "not a dataframe"
    y = pd.Series([0, 1])
    with pytest.raises(ValueError):
        prepare_data_for_modeling(X, y)