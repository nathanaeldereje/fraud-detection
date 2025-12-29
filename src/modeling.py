# src/modeling.py
import logging
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_logistic_regression(X_train, y_train, random_state=42):
    logger.info("Training Logistic Regression baseline...")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model

def train_xgboost(X_train, y_train, param_grid, cv=5, random_state=42):
    """
    Trains XGBoost using GridSearchCV and returns the full Grid object.
    """
    logger.info("Starting XGBoost hyperparameter tuning...")
    
    xgb = XGBClassifier(
        random_state=random_state,
        eval_metric='aucpr',
        scale_pos_weight=1 
    )
    
    grid = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=StratifiedKFold(n_splits=cv), 
        scoring='average_precision', 
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    logger.info(f"Best XGBoost Params: {grid.best_params_}")
    
    # RETURN THE FULL GRID OBJECT (not just the model)
    return grid