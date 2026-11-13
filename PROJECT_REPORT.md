# Customer Churn Analytics and Retention Support

## Abstract

This project develops a reproducible machine-learning workflow and interactive application for exploring customer churn in the IBM Telco Customer Churn dataset. Four classification pipelines are compared using five-fold stratified cross-validation on a training partition. The classifier with the highest mean F1 score for churn is then evaluated on an untouched test partition. The selected Logistic Regression model achieved a test F1 score of 0.614 and recall of 0.783 at a 0.50 threshold. A Streamlit application presents aggregate exploration, individual estimates, batch scoring, and model evaluation. Results are descriptive estimates from a historical sample and should not be interpreted as causal or production-ready decisions.

## 1. Problem statement

Customer attrition affects recurring revenue and service planning. The project investigates whether customer tenure, subscribed services, contract, and billing attributes can be used to identify customers whose historical profiles resemble churners. The intended output is an interpretable academic demonstration of a machine-learning lifecycle—not an automated decision about real customers.

### Research question

How effectively can standard classification algorithms distinguish churned from retained customers in the supplied telecom dataset, and what information does an interactive tool need to communicate those estimates responsibly?

## 2. Objectives

1. Summarize churn patterns and compare rates across customer segments.
2. Prepare mixed numeric and categorical customer data within a reusable preprocessing pipeline.
3. Compare four classifiers with churn-class F1 as the model-selection metric.
4. Evaluate the selected model on data not used for model selection.
5. Deliver a usable app for interactive exploration, individual estimation, and batch scoring.
6. Explain uncertainty, threshold behavior, and limitations alongside the results.

## 3. Dataset and target

The supplied CSV contains 7,043 customer records and 21 columns. `Churn` is the binary target (`Yes` = churned; `No` = stayed). There are 1,869 churned customers (26.5%) and 5,174 retained customers (73.5%). The identifier `customerID` is excluded from model input because it is not a customer service/account attribute.

The dataset includes demographic/account indicators, tenure, phone and internet services, contract type, payment method, and monthly/total charges. The `TotalCharges` field is converted to numeric; blank entries are treated as missing and imputed from training data.

## 4. Method

### Preprocessing

- Numeric values: median imputation followed by standard scaling.
- Categorical values: most-frequent imputation followed by one-hot encoding.
- Unknown categories at prediction time: ignored by the encoder.
- Preprocessing and classifier are packaged in a single scikit-learn pipeline.

### Experimental design

- Stratified train/test split: 80% training and 20% test, random seed 42.
- Candidate models: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting.
- Model selection: five-fold stratified cross-validation on the training partition only.
- Selection metric: mean F1 score for the churn class.
- Final comparison: each model is evaluated once on the untouched held-out test partition at threshold 0.50.

This split protects the final test results from being directly used to choose the model. It is still a single-sample evaluation; external validation and repeated experiments would be needed before drawing stronger conclusions.

### Metrics

- **Precision:** proportion of customers flagged as churners who churned.
- **Recall:** proportion of actual churners detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking ability across thresholds using true-positive and false-positive rates.
- **PR-AUC:** precision-recall ranking performance, particularly useful with class imbalance.
- **Accuracy:** overall fraction of correct classifications; interpreted alongside the other measures.

## 5. Results

At a 0.50 classification threshold, held-out test performance was:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 | 0.633 |
| Random Forest | 0.778 | 0.573 | 0.636 | 0.603 | 0.834 | 0.635 |
| Decision Tree | 0.740 | 0.507 | 0.789 | 0.617 | 0.830 | 0.615 |
| Gradient Boosting | 0.797 | 0.654 | 0.500 | 0.567 | 0.845 | 0.663 |

The Logistic Regression model was selected because it achieved the highest mean five-fold training cross-validation F1 (0.628). The held-out F1 values are reported for comparison but were not used to select the model. The Decision Tree's test F1 is close to the selected model; this difference underlines that model rankings can vary by split.

For the selected model, the held-out confusion counts are 747 true negatives, 288 false positives, 81 false negatives, and 293 true positives. Its recall is relatively high compared with precision at this threshold, meaning it identifies many churners but also flags retained customers. The appropriate balance depends on the cost and capacity of any follow-up process.

Observed churn rates in this sample are approximately 42.7% for month-to-month, 11.3% for one-year, and 2.8% for two-year contracts. These are associations only; the analysis does not show that changing a customer's contract would prevent churn.

## 6. Application

The Streamlit app provides:

- An overview of dataset totals, churn distribution, and contract-group rates.
- Individual profile entry with a probability estimate, configurable display threshold, and rule-based discussion prompts.
- Batch CSV scoring with required-column checks, optional customer identifiers, risk ranking, and downloadable results.
- A model performance page with held-out metrics, cross-validation results, confusion counts, and relative model signals.
- Filtered data exploration with charts, summary metrics, preview, and CSV export.

The app reports estimates and includes contextual warnings. It does not claim to explain causality or guarantee retention.

## 7. Conclusion

The tested models provide useful discrimination on this sample, with the selected Logistic Regression offering a recall-oriented balance at the demonstration threshold. The workflow demonstrates data preparation, cross-validated model selection, holdout evaluation, model serialization, and interactive deployment. The application also gives users ways to inspect and score records while keeping uncertainty and threshold effects visible.

## 8. Limitations and future work

- Evaluate on a later-time or independently sourced dataset to test generalization and drift.
- Use repeated nested cross-validation or additional samples to quantify selection variability more fully.
- Select thresholds using explicit intervention costs, capacity constraints, and validation data.
- Assess calibration and subgroup performance before any operational use.
- Test retention interventions experimentally before claiming they reduce churn.
- Expand automated coverage for malformed files, schema drift, and additional scoring behavior before production deployment.
- Establish privacy, access-control, monitoring, and retraining processes for any real deployment.

## 9. Reproducibility

From the project root, install dependencies, train the models, and launch the app:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

Training exports the selected fitted pipeline to `models/churn_model.joblib`, the model and evaluation report to `models/model_metrics.json`, and presentation charts to `outputs/`.
