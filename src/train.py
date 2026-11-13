"""Explore the Telco churn data, compare classifiers, and save the best pipeline."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
MODELS_FOLDER = PROJECT_ROOT / "models"


def make_pipeline(classifier, numeric_columns, categorical_columns):
    numeric_preprocessing = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_preprocessing = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numbers", numeric_preprocessing, numeric_columns),
        ("categories", categorical_preprocessing, categorical_columns),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Put the CSV in the project's data folder."
        )

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    MODELS_FOLDER.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    required_columns = {"Churn", "TotalCharges"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print("\nChurn distribution:")
    print(df["Churn"].value_counts().to_string())
    print("\nChurn percentage:")
    print((df["Churn"].value_counts(normalize=True).mul(100).round(1)).to_string())

    # Create presentation charts without blocking the training run with GUI windows.
    counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Stayed", "Churned"], counts.values, color=["steelblue", "tomato"])
    ax.set_title("Customer Churn Counts")
    ax.set_ylabel("Number of customers")
    ax.bar_label(bars)
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "churn_counts.png", dpi=150)
    plt.close(fig)

    df["ChurnFlag"] = df["Churn"].eq("Yes")
    contract_churn = (
        df.groupby("Contract", observed=False)["ChurnFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    print("\nChurn rate by contract (%):")
    print(contract_churn.round(1).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(contract_churn.index, contract_churn.values, color="darkorange")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Customers who churned (%)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "churn_by_contract.png", dpi=150)
    plt.close(fig)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=["Churn", "ChurnFlag", "customerID"], errors="ignore")
    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        invalid_labels = sorted(df.loc[y.isna(), "Churn"].astype(str).unique())
        raise ValueError(f"Unexpected churn labels in dataset: {', '.join(invalid_labels)}")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1500, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    results = {}
    pipelines = {}
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    threshold = 0.5
    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier, numeric_columns, categorical_columns)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        pipelines[name] = pipeline
        cv_summary = {}
        for metric in scoring:
            cv_summary[f"{metric}_mean"] = float(cv_scores[f"test_{metric}"].mean())
            cv_summary[f"{metric}_std"] = float(cv_scores[f"test_{metric}"].std())
        results[name] = {"cross_validation": cv_summary}

    # Select using training-only cross-validation, then evaluate once on the untouched holdout.
    best_name = max(
        results,
        key=lambda name: results[name]["cross_validation"]["f1_mean"],
    )
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(int)
        test_matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
        test_tn, test_fp, test_fn, test_tp = test_matrix.ravel()
        results[name].update({
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "pr_auc": float(average_precision_score(y_test, test_probabilities)),
            "confusion_matrix": {
                "true_negative": int(test_tn),
                "false_positive": int(test_fp),
                "false_negative": int(test_fn),
                "true_positive": int(test_tp),
            },
            "threshold": threshold,
        })

    best_model = pipelines[best_name]
    joblib.dump(best_model, MODELS_FOLDER / "churn_model.joblib")

    classifier = best_model.named_steps["classifier"]
    preprocessor = best_model.named_steps["preprocessing"]
    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    else:
        importance_values = abs(classifier.coef_[0])
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = sorted(
        [
            {"feature": str(feature), "importance": float(importance)}
            for feature, importance in zip(feature_names, importance_values)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:20]

    report = {
        "selected_model": best_name,
        "selection_metric": "Mean churn-class F1 across 5-fold stratified cross-validation",
        "positive_class": "Churned",
        "test_size": len(X_test),
        "train_size": len(X_train),
        "random_state": 42,
        "cv_folds": 5,
        "threshold": threshold,
        "feature_importance": feature_importance,
        "models": results,
    }
    (MODELS_FOLDER / "model_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    comparison = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ]
    print("\nModel comparison (positive class = churn):")
    print(comparison.round(3).to_string())
    print(
        f"\nSelected model by mean 5-fold CV churn F1: {best_name} "
        f"({results[best_name]['cross_validation']['f1_mean']:.3f})"
    )
    print("\nSelected model confusion matrix (rows=actual, columns=predicted; No then Yes):")
    print(confusion_matrix(y_test, (best_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)))
    print(f"\nSaved model: {MODELS_FOLDER / 'churn_model.joblib'}")
    print(f"Saved metrics: {MODELS_FOLDER / 'model_metrics.json'}")
    print(f"Saved charts in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
