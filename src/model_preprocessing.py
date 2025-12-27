# src/model_preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def prepare_data_for_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Dataset",
    imbalance_technique: str = "smote",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Complete preprocessing + imbalance handling pipeline with robust error handling.
    """
    try:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X must be DataFrame and y must be Series")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        print(f"\n=== Preparing {dataset_name} for Modeling ===")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        logger.info(f"Stratified split completed: Train {X_train.shape}, Test {X_test.shape}")
        
        # Preprocessor
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )
        
        logger.info(f"Preprocessor configured: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        feature_names = preprocessor.get_feature_names_out()
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        
        # Imbalance handling
        print(f"Applying {imbalance_technique.upper()}...")
        logger.info(f"Applying imbalance technique: {imbalance_technique}")
        
        if imbalance_technique == "smote":
            balancer = SMOTE(random_state=random_state)
        elif imbalance_technique == "undersample":
            balancer = RandomUnderSampler(random_state=random_state)
        elif imbalance_technique == "smotetomek":
            balancer = SMOTETomek(random_state=random_state)
        elif imbalance_technique == "none":
            logger.info("No resampling applied.")
            return X_train_processed, y_train, X_test_processed, y_test, preprocessor
        else:
            raise ValueError(f"Unknown imbalance_technique: {imbalance_technique}")
        
        X_train_bal, y_train_bal = balancer.fit_resample(X_train_processed, y_train)
        
        logger.info("Class distribution BEFORE balancing:")
        logger.info(pd.Series(y_train).value_counts(normalize=True).round(4).to_dict())
        logger.info("Class distribution AFTER balancing:")
        logger.info(pd.Series(y_train_bal).value_counts(normalize=True).round(4).to_dict())
        
        print(f"✅ Ready for modeling! Train Shape: {X_train_bal.shape}")
        return X_train_bal, y_train_bal, X_test_processed, y_test, preprocessor
    
    except Exception as e:
        logger.error(f"Error in prepare_data_for_modeling: {str(e)}")
        raise