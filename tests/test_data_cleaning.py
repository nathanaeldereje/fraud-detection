# tests/test_data_cleaning.py
import pandas as pd
import pytest
from src.data_cleaning import remove_duplicates, remove_missing_values

def test_remove_duplicates():
    # Test with duplicates
    df = pd.DataFrame({
        'user_id': [1, 1, 2, 3],
        'value': [10, 10, 20, 30]
    })
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 3
    assert cleaned.duplicated().sum() == 0

def test_remove_duplicates_no_dups():
    df = pd.DataFrame({'a': [1, 2, 3]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 3

def test_remove_duplicates_invalid_input():
    with pytest.raises(ValueError):
        remove_duplicates("not a dataframe")

def test_remove_missing_values():
    df = pd.DataFrame({
        'a': [1, None, 3],
        'b': [4, 5, None]
    })
    cleaned = remove_missing_values(df)
    assert len(cleaned) == 1
    assert cleaned.isnull().sum().sum() == 0

def test_remove_missing_values_no_missing():
    df = pd.DataFrame({'a': [1, 2, 3]})
    cleaned = remove_missing_values(df)
    assert len(cleaned) == 3