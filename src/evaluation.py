# src/evaluation.py
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, dataset_name, model_name):
    """
    Evaluates model and returns a dictionary of metrics for comparison.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"\n=== {dataset_name} - {model_name} Evaluation ===")
    print(f"AUC-PR: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot PR Curve (Optional: Can comment out if cluttering)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.title(f'PR Curve: {dataset_name} - {model_name}')
    plt.legend()
    plt.show()

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "AUC-PR": pr_auc,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec
    }