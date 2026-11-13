# Customer Churn Analytics and Retention Support

A reproducible end-semester data science project that explores telecom churn, compares classification models, estimates customer-level churn probability, and supports single-customer or batch review through a Streamlit application.

> **Scope:** This is an academic decision-support demonstration. Scores and follow-up prompts are estimates and discussion aids, not guarantees, causal recommendations, or a validated production decision system.

## Project highlights

- Interactive overview and dataset exploration with segment filters, summary metrics, and filtered-data export.
- Individual churn estimation with an adjustable display threshold and transparent follow-up prompts.
- Batch scoring for customer CSV files, ranked results, and downloadable probability/label output.
- Four end-to-end scikit-learn pipelines compared using five-fold stratified cross-validation.
- Model selection on training-only cross-validation, followed by one evaluation on a separate stratified test split.
- Evaluation dashboard for precision, recall, F1, ROC-AUC, PR-AUC, confusion counts, cross-validation variability, and model signals.
- Reproducible training script that saves the selected pipeline, metrics, and presentation charts.

## Project question

Can customer account, service, billing, and tenure attributes help identify customers with elevated historical churn risk, and how do common classification models compare when churners are the positive class?

The project studies associations in a public sample. It does not establish why a customer leaves or whether any retention action changes that outcome.

## Dataset

The included IBM Telco Customer Churn CSV contains 7,043 customer records. The target is `Churn` (`Yes` = churned, `No` = stayed): 1,869 churned (26.5%) and 5,174 stayed (73.5%). The customer identifier is excluded from training. `TotalCharges` is converted to numeric, with missing values handled inside the fitted preprocessing pipeline.

The application expects the supplied file at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Methodology

1. Read the dataset and separate the binary target from model features.
2. Convert `TotalCharges` to numeric. Impute numeric missing values with the training median, scale numeric features, impute categorical values with the most frequent training value, and one-hot encode categories. Unknown categories are ignored at inference.
3. Create a stratified 80/20 train/test split using random seed 42.
4. Compare Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting with five-fold stratified cross-validation on the training partition. Candidate selection uses mean churn-class F1 only; the held-out test set is not used for selection.
5. Evaluate each candidate on the untouched test partition at a 0.50 threshold. Save the cross-validation and test metrics, confusion counts, and selected pipeline.

The trained preprocessing and classifier are saved together as one pipeline to avoid a separate, potentially inconsistent inference transform. The app's adjustable threshold changes the high/low display label, not the model probability.

## Run locally (Windows PowerShell)

Tested with Python 3.13. Dependencies are pinned in `requirements.txt` for repeatable setup.

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Keep the terminal open while using the app; press **Ctrl+C** to stop it.

To use an existing trained model without retraining, run only `streamlit run app.py`. If the model file is missing, train it first.

## App pages

- **Overview:** Dataset totals, churn distribution, contract comparison, and current model snapshot.
- **Estimate a customer:** Enter an account profile and receive a probability, threshold-based label, and rule-based discussion prompts.
- **Batch scoring:** Upload a CSV with all model input columns. `customerID` is optional; extra columns (including `Churn`) are ignored. Results include churn probability and risk level; the downloadable CSV contains every scored row.
- **Model performance:** Compare the held-out test metrics, see training-only cross-validation results, inspect confusion counts, and review selected-model signals.
- **Explore the data:** Filter the sample by contract, internet service, payment method, and churn outcome; inspect aggregates and export the filtered records.

On the **Batch scoring** page, expand **Required CSV columns and template** to see the exact input schema and download a blank CSV template.

## Verify the application

Run the included Streamlit smoke and behavior tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

## Artifacts

```text
app.py                              Streamlit application
src/train.py                        Data preparation, training, evaluation, and export
models/churn_model.joblib           Fitted preprocessing/model pipeline
models/model_metrics.json           Cross-validation and held-out test report
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
outputs/churn_counts.png            Generated churn distribution chart
outputs/churn_by_contract.png       Generated contract segment chart
PROJECT_REPORT.md                   Presentation-ready project report
requirements.txt                    Python dependencies
```

Run `python src/train.py` after changing the dataset or training code to regenerate the model, report metrics, and charts.

## Data and decision limitations

- The evaluation uses one fixed random split of one historical sample; it is not evidence of future or out-of-population performance.
- The dataset is imbalanced. Accuracy alone can obscure churner detection, so precision, recall, F1, ROC-AUC, and PR-AUC are also reported.
- A 0.50 threshold is a demonstration default, not a cost-optimized operating point.
- Model signals describe how the selected model uses the data; they do not imply causation or explain an individual prediction.
- The follow-up prompts are transparent rules, not experimentally validated interventions.
- Do not upload sensitive or identifiable customer information to an unapproved environment. This demo is not designed as a secure production service.

See [PROJECT_REPORT.md](./PROJECT_REPORT.md) for the full academic-style summary, results, and discussion.
