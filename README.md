# Improved Detection of Fraud Cases for E-Commerce and Bank Transactions
**Adey Innovations Inc. – Fraud Detection Project**  
*December 2025*

An end-to-end machine learning project to develop accurate and robust fraud detection models for e-commerce and bank credit card transactions. The models incorporate geolocation analysis, transaction velocity patterns, and advanced ensemble techniques while addressing severe class imbalance and providing explainable insights via SHAP.

---
## Business Goal
Adey Innovations Inc., a leader in financial technology, seeks to deliver cutting-edge fraud detection solutions that:

- **Accurately identify fraudulent transactions** in real-time to minimize financial losses for e-commerce and banking clients.  
- **Reduce false positives** to maintain excellent customer experience and avoid alienating legitimate users.  
- **Leverage geolocation and behavioral signals** (e.g., time between signup and purchase, transaction frequency) for stronger fraud detection.  
- **Provide transparent, interpretable models** that build trust with customers, partners, and regulators.

A successful system will enable faster risk mitigation, lower fraud-related losses, and stronger confidence in digital transactions.

## Project Overview
This project analyzes two highly imbalanced datasets:  
1. E-commerce transactions (`Fraud_Data.csv`) with rich user, device, IP, and timing details.  
2. Anonymized bank credit card transactions (`creditcard.csv`) with PCA-transformed features.

We perform thorough EDA, geolocation merging, feature engineering, imbalance handling (SMOTE and SMOTETomek), model training, and SHAP-based explainability to derive actionable business rules.

### Key Deliverables
- [x] Comprehensive EDA with visualizations of fraud patterns and class imbalance  
- [x] Geolocation integration (IP → Country) and rich feature engineering (velocity, time-since-signup, etc.)  
- [x] Resampling strategy for severe class imbalance (SMOTE for e-commerce, SMOTETomek for credit card)  
- [x] Baseline Logistic Regression + advanced ensemble (Random Forest/XGBoost/LightGBM)  
- [x] Evaluation using AUC-PR, F1-Score, Precision-Recall curves, and Confusion Matrix  
- [x] SHAP global/local explanations with business recommendations  
- [x] Clean, organized, and reproducible repository

---
## Fraud Detection Business Understanding
### Key Challenges in Fraud Detection
1. **Extreme Class Imbalance**  
   Fraudulent transactions typically represent <<1% of total volume. Standard accuracy is misleading — we prioritize **Precision-Recall AUC** and **F1-score**.

2. **Security vs. Customer Experience Trade-off**  
   - High recall → catches more fraud but risks more false positives (customer friction).  
   - High precision → fewer false alarms but may miss real fraud (financial loss).  
   **Strategy:** Tune models and thresholds to optimize business cost (false negative cost >> false positive cost).

3. **Need for Explainability**  
   Financial institutions require interpretable decisions for regulatory compliance, audit trails, and deriving actionable rules (e.g., "flag transactions from high-risk countries within 1 hour of signup").

4. **Real-Time Requirements**  
   Models must be fast and deployable for live scoring while incorporating contextual signals like device reuse and geolocation anomalies.

---
## Project Structure
```text
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── __init__.py
│   ├── eda-creditcard.ipynb
│   ├── eda-fraud-data.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── README.md
│   └── shap-explainability.ipynb
├── outputs/
│   └── shap/
│       ├── false_negative_missed_fraud_2.html
│       ├── false_positive_legitimate_flagged_as_fraud_310.html
│       └── true_positive_correctly_detected_fraud_17.html
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── data_preprocessing.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── model_preprocessing.py
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   └── test_model_preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
```
---
## Tech Stack
- Core: Python 3.12, pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- Machine Learning: scikit-learn, imbalanced-learn, xgboost, lightgbm
- Explainability: shap
- Utilities: joblib, jupyter
---
## Quick Start
```bash
# Clone the repository
git clone https://github.com/nathanaeldereje/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
Place the raw datasets in `data/raw/`:

* `Fraud_Data.csv`
* `IpAddress_to_Country.csv`
* `creditcard.csv`

Run the preprocessing pipeline:
```bash
   python scripts/preprocess.py
```

Explore and build the project using the Jupyter notebooks in order:
1. `eda-fraud-data.ipynb`
2. `eda-creditcard.ipynb`
3. `feature-engineering.ipynb`
4. `modeling.ipynb`
5. `shap-explainability.ipynb`
---
## Model Comparison Summary

| Dataset       | Model                | AUC-PR  | Fraud Precision | Fraud Recall | F1 (Fraud) | Notes |
|---------------|----------------------|---------|-----------------|--------------|------------|-------|
| Fraud_Data    | Logistic Regression  | 0.393   | 0.17            | 0.70         | 0.28       | High recall, low precision |
| Fraud_Data    | XGBoost (tuned)      | **0.607** | **0.96**      | 0.53         | **0.68**   | **Selected** — high precision, stable |
| CreditCard    | Logistic Regression  | 0.713   | 0.05            | 0.87         | 0.10       | Many false positives |
| CreditCard    | XGBoost (tuned)      | **0.814** | **0.76**      | **0.82**     | **0.79**   | **Selected** — excellent balance |
---


## Current Progress (as of December 29, 2025)
| Task | Status | Notes |
| :--- | :--- | :--- |
| **Data Loading & Initial Cleaning** | ✅ Completed | Raw datasets loaded and inspected |
| **EDA (Both Datasets)** | ✅ Completed | Visualizations and imbalance analysis ongoing |
| **Geolocation Merging & Feature Engineering** | ✅ Completed | IP-to-country + velocity features next |
| **Data Transformation & Imbalance Handling** | ✅ Completed | Scaling/encoding pipeline + SMOTE/SMOTETomek on training only |
| **Model Building & Evaluation** | ✅ Completed | Logistic baseline + tuned XGBoost |
| **SHAP Explainability & Recommendations** | ✅ Completed | Global/local analysis + actionable rules |
---
Challenge completed – December 29, 2025 
Built by Nathanael Dereje