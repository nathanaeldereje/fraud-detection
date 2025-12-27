# tests/test_feature_engineering.py
import pandas as pd
import pytest
from src.data_preprocessing import engineer_features

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'user_id': [100, 100, 200],
        'signup_time': ['2025-01-01 10:00:00', '2025-01-01 12:00:00', '2025-01-02 08:00:00'],
        'purchase_time': ['2025-01-02 15:00:00', '2025-01-03 09:00:00', '2025-01-02 14:00:00'],
        'purchase_value': [50, 75, 100],
        'device_id': ['D1', 'D1', 'D2']
    })

def test_engineer_features(sample_df):
    result = engineer_features(sample_df)
    
    # Check new columns exist
    expected_cols = ['hour_of_day', 'day_of_week', 'time_since_signup', 
                     'user_txn_count', 'user_total_spent', 'user_avg_purchase']
    for col in expected_cols:
        assert col in result.columns
    
    # Check calculations
    assert result.loc[0, 'hour_of_day'] == 15
    assert result.loc[0, 'time_since_signup'] == 29.0  # ~29 hours
    assert result.loc[0, 'user_txn_count'] == 2
    assert result.loc[0, 'user_total_spent'] == 125
    assert result.loc[0, 'user_avg_purchase'] == 62.5

def test_engineer_features_missing_column(sample_df):
    df_missing = sample_df.drop(columns=['purchase_value'])
    with pytest.raises(ValueError, match="Missing required columns"):
        engineer_features(df_missing)