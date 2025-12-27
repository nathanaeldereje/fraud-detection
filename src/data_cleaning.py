# src/data_cleaning.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from a DataFrame and logs a summary report.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
        
    Raises:
        ValueError: If input is not a pandas DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        initial_rows = len(df)
        num_duplicates = df.duplicated().sum()
        
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        final_rows = len(df_cleaned)
        
        logger.info(f"Initial Row Count: {initial_rows:,}")
        logger.info(f"Duplicate Rows Found: {num_duplicates:,}")
        logger.info(f"Rows After Cleaning: {final_rows:,}")
        
        if initial_rows != final_rows:
            logger.info(f"✅ Removed {initial_rows - final_rows:,} duplicate rows.")
        else:
            logger.info("✅ No duplicates found.")
            
        return df_cleaned
    
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {str(e)}")
        raise


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with missing values from the DataFrame and logs a summary.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with missing values removed.
        
    Raises:
        ValueError: If input is not a pandas DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        initial_rows = len(df)
        df_cleaned = df.dropna().reset_index(drop=True)
        remaining_rows = len(df_cleaned)
        removed = initial_rows - remaining_rows
        
        total_nulls_after = df_cleaned.isnull().sum().sum()
        
        if total_nulls_after == 0:
            logger.info("✅ No missing values detected. Dataset is clean.")
            if removed > 0:
                logger.info(f"   (Removed {removed:,} rows containing null values)")
        else:
            logger.warning(f"⚠️ Warning: {total_nulls_after:,} missing values remain after dropna().")
        
        return df_cleaned
    
    except Exception as e:
        logger.error(f"Error in remove_missing_values: {str(e)}")
        raise