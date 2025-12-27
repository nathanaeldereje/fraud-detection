# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import logging


# Initialize logger for this module
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering operations to the cleaned Fraud_Data dataframe.
    
    Creates:
        - hour_of_day: hour of purchase (0-23)
        - day_of_week: weekday of purchase (0=Monday, 6=Sunday)
        - time_since_signup: hours between signup and purchase
        - user_txn_count: total transactions per user
        - user_total_spent: total purchase value per user
        - user_avg_purchase: average per transaction
    
    Args:
        df (pd.DataFrame): Cleaned dataframe with required columns.
    
    Returns:
        pd.DataFrame: DataFrame with new engineered features.
        
    Raises:
        ValueError: If required columns are missing or date parsing fails.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        required_cols = ['signup_time', 'purchase_time', 'user_id', 'purchase_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        
        # Safe datetime conversion
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        
        if df['signup_time'].isnull().any() or df['purchase_time'].isnull().any():
            raise ValueError("Failed to parse dates in 'signup_time' or 'purchase_time' — check data format")
        
        # Time-based features
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.weekday
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Velocity features
        df['user_txn_count'] = df.groupby('user_id')['user_id'].transform('count')
        df['user_total_spent'] = df.groupby('user_id')['purchase_value'].transform('sum')
        df['user_avg_purchase'] = df['user_total_spent'] / df['user_txn_count']
        
        # Drop original identifiers and timestamps (as intended)
        cols_to_drop = ['user_id', 'device_id', 'signup_time', 'purchase_time']
        dropped = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=dropped, inplace=True)
        
        logger.info("✅ Feature engineering completed successfully.")
        logger.info("Sample of new features:\n%s", 
                    df[['hour_of_day', 'day_of_week', 'time_since_signup', 
                        'user_txn_count', 'user_total_spent', 'user_avg_purchase']].head().to_string())
        
        return df
    
    except Exception as e:
        logger.error(f"Error in engineer_features: {str(e)}")
        raise



def build_preprocessor(df: pd.DataFrame):
    """
    Builds a sklearn ColumnTransformer for scaling numeric + one-hot encoding categorical features.
    Excludes target column ('class' or 'Class').
    """
    # Auto-detect feature types (exclude target)
    target_candidates = ['class', 'Class']
    target = next((col for col in target_candidates if col in df.columns), None)

    feature_df = df.drop(columns=[target]) if target else df

    numeric_features = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor

