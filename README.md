# Customer Churn Prediction with Explainable AI

Predicts which customers of a subscription service are likely to churn
and shows why, using SHAP and LIME. Includes a Streamlit dashboard.

## Features

- Loads the Telco Customer Churn dataset, or generates a synthetic dataset
  with the same schema if no CSV is present.
- Engineered features: average charges per month, charge-to-tenure ratio,
  number of add-ons, tenure bucket.
- SMOTE for class imbalance, applied inside an imblearn Pipeline so it only
  runs on training folds.
- Three classifiers: Logistic Regression, Random Forest, XGBoost. Picks the
  best by ROC-AUC.
- Evaluation: ROC, Precision-Recall, confusion matrix, F1, 5-fold CV ROC-AUC.
- Explanations: global SHAP summary, per-customer SHAP drivers, LIME.
- 5-page Streamlit dashboard.

## Project layout

```
.
├── main.py                  CLI: data | train | evaluate | explain | all
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── app/
│   └── streamlit_app.py
├── data/
├── models/
└── reports/
```

## Setup

```bash
pip install -r requirements.txt
python main.py all
streamlit run app/streamlit_app.py
```

Individual stages:

```bash
python main.py data
python main.py train
python main.py evaluate
python main.py explain
```

## Using the real Telco dataset

Drop the Kaggle CSV at `data/telco_churn.csv`. The loader uses it
automatically. If missing, a synthetic dataset of the same schema is
generated so the pipeline still runs.

## Dashboard pages

1. Portfolio Overview - totals, churn probability distribution, churn by
   contract type, tenure vs charges scatter.
2. Model Performance - metrics table for all three models, plus ROC, PR,
   confusion matrix, and feature importance plots.
3. High-Risk Customers - table of customers with churn probability >= 0.60,
   downloadable as CSV.
4. Score a Customer - pick an existing customer or build one manually. Shows
   probability, risk tier, SHAP top drivers, and LIME breakdown.
5. Global Explainability - SHAP summary plot.

## Notes

- SMOTE runs inside the imblearn Pipeline so cross-validation doesn't leak
  oversampled rows into the validation fold.
- TreeExplainer is used for the tree models, LinearExplainer for logistic
  regression.
- The background sample for SHAP/LIME is persisted as CSV (not pickled) so
  artifacts move cleanly across Python environments.
