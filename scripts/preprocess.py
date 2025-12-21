# scripts/preprocess.py
import sys
import pandas as pd


import os

# Get the absolute path of the project root directory (one level up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_and_clean_fraud_data(
    fraud_path='data/raw/Fraud_Data.csv',
    ip_path='data/raw/IpAddress_to_Country.csv'
) -> pd.DataFrame:
    """
    Full cleaning + feature engineering for Fraud_Data.csv
    """
    from src.data_cleaning import remove_duplicates, remove_missing_values
    from src.data_processing import map_ips_to_countries
    from src.data_preprocessing import engineer_features

    df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)

    print("Starting Fraud_Data preprocessing...")

    df = remove_duplicates(df)
    df = remove_missing_values(df)
    df = map_ips_to_countries(df, ip_df)
    df = engineer_features(df)

    print("✅ Fraud_Data preprocessing complete.")
    return df


def load_and_clean_creditcard_data(path='data/raw/creditcard.csv') -> pd.DataFrame:
    """
    Minimal cleaning for creditcard.csv (usually very clean)
    """
    df = pd.read_csv(path)

    print("Starting creditcard.csv preprocessing...")
    print(f"Initial shape: {df.shape}")

    # Check and remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - df.shape[0]} duplicate rows.")

    # Check missing values
    if df.isnull().sum().sum() == 0:
        print("✅ No missing values found.")
    else:
        print("Dropping rows with missing values...")
        df = df.dropna()

    print(f"Final shape: {df.shape}")
    print("✅ creditcard.csv preprocessing complete.")
    return df



if __name__ == "__main__":
    # Allow running from command line for testing
    print("Preprocessing Fraud_Data...")
    fraud_df = load_and_clean_fraud_data()
    fraud_df.to_csv('data/processed/fraud_data_engineered.csv', index=False)

    print("\nPreprocessing creditcard...")
    cc_df = load_and_clean_creditcard_data()
    cc_df.to_csv('data/processed/creditcard_processed.csv', index=False)

    print("\n✅ All datasets processed and saved to data/processed/")