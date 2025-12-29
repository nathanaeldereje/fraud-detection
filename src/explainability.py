# src/explainability.py
import os
import re
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from IPython.display import display 

def get_feature_names(preprocessor):
    """Extracts feature names from ColumnTransformer"""
    output_features = []
    
    # Loop through transformers
    for name, pipe, features in preprocessor.transformers_:
        if name == 'remainder':
            continue
            
        # If the feature list is empty (e.g. no categorical cols in CreditCard), skip
        if hasattr(features, "__len__") and len(features) == 0:
            continue
            
        if hasattr(pipe, 'get_feature_names_out'):
            try:
                # Try to get names (works if fitted)
                output_features.extend(pipe.get_feature_names_out(features))
            except (NotFittedError, ValueError):
                # If the transformer wasn't fitted (unused), just skip it
                pass
            except TypeError:
                # Fallback for older sklearn versions or specific cases
                try:
                    output_features.extend(pipe.get_feature_names_out())
                except:
                    pass
        else:
            # For transformers without get_feature_names_out (e.g. simple passthrough)
            output_features.extend(features)
            
    return output_features


def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=10):
    """
    Plots built-in feature importance from XGBoost.
    Args:
        model: Trained model object
        feature_names: List of feature names
        title: Title for the plot (string)
        top_n: Number of top features to show (int)
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlim([-1, top_n])
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not support built-in feature importance.")

def get_prediction_indices(y_true, y_pred):
    """
    Returns indices for TP, TN, FP, FN.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    return {
        'TP': tp_indices, 
        'TN': tn_indices, 
        'FP': fp_indices, 
        'FN': fn_indices
    }


def plot_shap_force(idx, explainer, data_df, title, save_dir="../outputs/shap"):
    """
    Plots and saves a SHAP force plot for a specific observation.
    
    Args:
        idx (int): Index of the observation to explain.
        explainer: The SHAP explainer object.
        data_df (pd.DataFrame): The feature data (must be a DataFrame with column names).
        title (str): Title to print before the plot.
        save_dir (str): Directory to save the HTML file (relative to notebook).
    """
    print(f"--- {title} (Index: {idx}) ---")

    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame.")

    # 1. Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # 2. Calculate SHAP values
    shap_val_single = explainer.shap_values(data_df.iloc[[idx]])

    # 3. Generate the interactive plot
    plot = shap.force_plot(
        explainer.expected_value,
        shap_val_single[0],
        data_df.iloc[[idx]],
        matplotlib=False,
        link="logit"
    )

    # 4. Create a safe filename (CRITICAL for Windows)
    # Replaces ":", spaces, and other special chars with "_"
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '_')
    filename = f"{safe_title}_{idx}.html"
    filepath = os.path.join(save_dir, filename)

    # 5. Save and Display
    try:
        shap.save_html(filepath, plot)
        print(f"✅ SHAP plot saved to: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

    display(plot)
