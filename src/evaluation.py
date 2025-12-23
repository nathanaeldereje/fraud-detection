# src/evaluation.py
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, dataset_name: str):
    """
    Evaluates a trained model on test data using metrics suitable for imbalanced fraud detection.
    
    Prints:
    - AUC-PR (primary metric)
    - Classification report (Precision, Recall, F1 per class)
    - Confusion matrix
    - Plots Precision-Recall curve
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    print(f"\n=== {dataset_name} Evaluation Results ===")
    print(f"AUC-PR: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.show()