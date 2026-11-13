from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_FILE = PROJECT_ROOT / "models" / "model_metrics.json"
DATA_FILE = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(
    page_title="Customer Churn | Retention Studio",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e7ebf2; padding: 18px 20px;
        border-radius: 14px; box-shadow: 0 2px 10px rgba(20, 35, 60, .04);
    }
    .hero {
        padding: 28px 32px; border-radius: 18px; margin-bottom: 22px;
        background: linear-gradient(115deg, #102a43 0%, #174c70 65%, #287a83 100%);
        color: white;
    }
    .hero h1 { color: white; margin: 0 0 8px 0; font-size: 2.15rem; }
    .hero p { color: #e4f1f5; margin: 0; font-size: 1rem; }
    .section-note { color: #53657a; font-size: .94rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📡 Retention Studio")
    st.caption("Customer churn analysis and risk estimation")
    st.divider()
    page = st.radio(
        "Go to",
        ["Overview", "Estimate a customer", "Batch scoring", "Model performance", "Explore the data"],
    )
    st.divider()
    st.caption("Academic demonstration using the IBM Telco Customer Churn dataset.")


@st.cache_resource
def load_model(path: str, modified_time: float):
    return joblib.load(path)


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: str, modified_time: float):
    with open(path, encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def retention_suggestions(customer: dict) -> list[tuple[str, str]]:
    """Return transparent follow-up ideas based on the entered profile."""
    suggestions = []
    if customer["Contract"] == "Month-to-month":
        suggestions.append(("Review contract fit", "Ask whether a longer-term plan would suit their needs."))
    if customer["MonthlyCharges"] >= 90:
        suggestions.append(("Review plan and charges", "Check whether the current plan matches the services they use."))
    if customer["tenure"] < 12:
        suggestions.append(("Support early tenure", "Offer onboarding help and ask about any service issues."))
    if customer["TechSupport"] == "No":
        suggestions.append(("Check support needs", "Ask whether the available technical support options meet their needs."))
    if customer["PaymentMethod"] == "Electronic check":
        suggestions.append(("Review payment experience", "Check whether the customer has had payment or billing difficulties."))
    if not suggestions:
        suggestions.append(("Stay engaged", "Continue regular service check-ins and monitor changes in customer needs."))
    return suggestions


if not MODEL_FILE.exists():
    st.error("The trained model is missing. From the project folder, run `python src/train.py` first.")
    st.stop()

try:
    model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

data = None
data_load_error = None
if DATA_FILE.exists():
    try:
        data = load_data(str(DATA_FILE))
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        data_load_error = str(exc)

metrics = None
if METRICS_FILE.exists():
    try:
        metrics = load_metrics(str(METRICS_FILE), METRICS_FILE.stat().st_mtime)
    except (OSError, json.JSONDecodeError):
        metrics = None


if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>Customer churn, made actionable</h1>
            <p>Explore historical churn patterns, estimate an individual profile, and review practical follow-up ideas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if data is not None and "Churn" in data.columns:
        total_customers = len(data)
        churned = int(data["Churn"].eq("Yes").sum())
        churn_rate = (churned / total_customers * 100) if total_customers else 0
        stayed = total_customers - churned
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers in dataset", f"{total_customers:,}")
        c2.metric("Historical churn rate", f"{churn_rate:.1f}%")
        c3.metric("Customers retained", f"{stayed:,}")

        if metrics:
            selected_name = metrics.get("selected_model", "Model")
            selected_scores = metrics.get("models", {}).get(selected_name, {})
            st.subheader("Current model snapshot")
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected model", selected_name)
            m2.metric("Churn recall", f"{selected_scores.get('recall', 0):.1%}")
            m3.metric("Churn F1 score", f"{selected_scores.get('f1', 0):.1%}")
            st.caption("Scores are from one held-out test split. Open Model performance for the full comparison.")

        left, right = st.columns([1.3, 1])
        with left:
            st.subheader("Churn by contract")
            if "Contract" in data.columns:
                by_contract = (
                    data.assign(Churned=data["Churn"].eq("Yes"))
                    .groupby("Contract", observed=False)["Churned"]
                    .mean()
                    .mul(100)
                    .sort_values(ascending=False)
                )
                st.bar_chart(by_contract, y_label="Churn rate (%)")
                st.caption("Percent of customers in each contract group who churned in this dataset.")
        with right:
            st.subheader("Stayed vs churned")
            outcome_counts = data["Churn"].value_counts().rename(index={"No": "Stayed", "Yes": "Churned"})
            st.bar_chart(outcome_counts, y_label="Customers")
            st.caption(f"{churned:,} churned · {stayed:,} stayed")
    else:
        if data_load_error:
            st.warning(f"The dataset could not be read: {data_load_error}")
        else:
            st.info("Dataset CSV not found. Prediction still works; add the CSV under the project's `data` folder to show overview charts.")

    st.subheader("How to use this demo")
    st.markdown(
        "1. Open **Estimate a customer** and enter the customer's service and account details.\n"
        "2. Review the estimated churn probability and the follow-up ideas.\n"
        "3. Use **Explore the data** to inspect aggregate patterns in the sample dataset."
    )
    st.warning("A risk score is an estimate from historical data. It is not a guarantee that a customer will leave or that an action will retain them.")


elif page == "Estimate a customer":
    st.title("Estimate churn risk")
    st.markdown('<p class="section-note">Enter a customer profile. The model returns an estimated probability of churn.</p>', unsafe_allow_html=True)

    default_threshold = float(metrics.get("threshold", 0.5)) if metrics else 0.5
    threshold = st.slider(
        "Flag as higher risk at or above",
        min_value=0.10,
        max_value=0.90,
        value=min(max(default_threshold, 0.10), 0.90),
        step=0.05,
        format="%.0f%%",
        help="Changing this display threshold changes the label, not the model's estimated probability.",
    )

    with st.form("customer_details"):
        st.subheader("Account and household")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with a2:
            senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
        with a3:
            partner = st.selectbox("Has a partner", ["Yes", "No"])
        with a4:
            dependents = st.selectbox("Has dependents", ["Yes", "No"])

        st.subheader("Services")
        b1, b2, b3 = st.columns(3)
        with b1:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        with b2:
            online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        with b3:
            tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

        st.subheader("Billing and tenure")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        with c3:
            payment_method = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with c4:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

        d1, d2 = st.columns(2)
        with d1:
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=5.0)
        with d2:
            total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=840.0, step=50.0)

        submitted = st.form_submit_button("Estimate churn risk", type="primary", width="stretch")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        input_frame = pd.DataFrame([customer])
        try:
            probability = float(model.predict_proba(input_frame)[0, 1])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        st.divider()
        st.subheader("Prediction")
        result_col, detail_col = st.columns([1, 2])
        with result_col:
            st.metric("Estimated churn probability", f"{probability:.1%}")
            st.progress(probability, text=f"Risk score: {probability:.0%}")
        with detail_col:
            if probability >= threshold:
                st.warning(f"Higher estimated risk using the selected {threshold:.0%} threshold.")
            else:
                st.success(f"Lower estimated risk using the selected {threshold:.0%} threshold.")
            st.caption("The threshold changes the high/low label, not the probability. It is a demonstration setting; a real use case should choose it based on intervention costs and validation results.")

        st.subheader("Possible retention follow-ups")
        for title, detail in retention_suggestions(customer):
            st.markdown(f"**{title}**  \n{detail}")
        st.caption("These are rule-based discussion prompts, not guaranteed interventions or causal recommendations.")


elif page == "Batch scoring":
    st.title("Score a customer list")
    st.markdown(
        '<p class="section-note">Upload a CSV containing the model input columns to score multiple customer profiles at once.</p>',
        unsafe_allow_html=True,
    )
    st.caption("The customerID column is optional and is carried into the results when provided. Extra columns, including an existing Churn label, are ignored.")
    expected_features = list(model.feature_names_in_)
    with st.expander("Required CSV columns and template"):
        st.code(", ".join(expected_features), language="text")
        st.download_button(
            "Download blank CSV template",
            data=pd.DataFrame(columns=expected_features).to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_input_template.csv",
            mime="text/csv",
        )
    batch_threshold = st.slider(
        "Flag as higher risk at or above",
        min_value=0.10,
        max_value=0.90,
        value=min(max(float(metrics.get("threshold", 0.5)) if metrics else 0.5, 0.10), 0.90),
        step=0.05,
        format="%.0f%%",
        key="batch_threshold",
    )
    uploaded_file = st.file_uploader("Choose a customer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()

        if batch_data.empty:
            st.error("The uploaded CSV has no customer rows.")
            st.stop()

        missing_features = [feature for feature in expected_features if feature not in batch_data.columns]
        if missing_features:
            st.error(
                "The CSV is missing required model inputs: "
                + ", ".join(missing_features)
            )
            st.stop()

        scoring_data = batch_data.loc[:, expected_features].copy()
        if "TotalCharges" in scoring_data.columns:
            raw_total_charges = scoring_data["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
            numeric_total_charges = pd.to_numeric(raw_total_charges, errors="coerce")
            invalid_total_charges = raw_total_charges.notna() & numeric_total_charges.isna()
            if invalid_total_charges.any():
                invalid_rows = (invalid_total_charges[invalid_total_charges].index + 2).tolist()
                st.error(
                    "TotalCharges must be numeric. Invalid values were found on CSV rows: "
                    + ", ".join(map(str, invalid_rows[:10]))
                    + (" …" if len(invalid_rows) > 10 else "")
                )
                st.stop()
            scoring_data["TotalCharges"] = numeric_total_charges

        try:
            churn_probability = model.predict_proba(scoring_data)[:, 1]
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Could not score these customer rows: {exc}")
            st.stop()

        scored_data = batch_data.drop(columns=["Churn"], errors="ignore").copy()
        scored_data["Churn probability"] = churn_probability
        scored_data["Risk level"] = [
            "Higher" if probability >= batch_threshold else "Lower"
            for probability in churn_probability
        ]
        high_risk_count = int((churn_probability >= batch_threshold).sum())
        summary = st.columns(3)
        summary[0].metric("Customers scored", f"{len(scored_data):,}")
        summary[1].metric("Higher-risk customers", f"{high_risk_count:,}")
        summary[2].metric("Average churn probability", f"{churn_probability.mean():.1%}")

        st.dataframe(
            scored_data.sort_values("Churn probability", ascending=False).head(100),
            width="stretch",
        )
        st.caption(f"Showing the 100 highest-risk rows at most; download contains all {len(scored_data):,} scored customers.")
        st.download_button(
            "Download scored customers",
            data=scored_data.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_scores.csv",
            mime="text/csv",
            width="stretch",
        )
        st.warning("Scores are estimates based on the supplied model. They are not guarantees or proven retention recommendations.")


elif page == "Model performance":
    st.title("Model performance")
    st.markdown(
        '<p class="section-note">Four classifiers were compared using stratified cross-validation on the training data. The selected model was then evaluated once on an untouched test split.</p>',
        unsafe_allow_html=True,
    )
    if not metrics:
        st.info("Model metrics are not available yet. Run `python src/train.py` to compare models and save the report.")
    else:
        st.success(
            f"Selected model: **{metrics.get('selected_model', 'Unknown')}** "
            f"(selected by churn-class {metrics.get('selection_metric', 'F1 score')})."
        )
        st.caption(
            f"Training rows: {metrics.get('train_size', '—'):,} · "
            f"Test rows: {metrics.get('test_size', '—'):,} · "
            f"Positive class: {metrics.get('positive_class', 'Churned')} · "
            f"Classification threshold: {metrics.get('threshold', 0.5):.0%}"
        )

        models = metrics.get("models", {})
        if models:
            comparison = pd.DataFrame(models).T[
                ["precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"]
            ]
            comparison.index.name = "Model"
            st.subheader("Side-by-side comparison")
            st.bar_chart(comparison[["precision", "recall", "f1", "roc_auc", "pr_auc"]], y_label="Score")
            st.dataframe(
                comparison.style.format("{:.3f}"),
                width="stretch",
            )
            cv_comparison = pd.DataFrame({
                name: result.get("cross_validation", {})
                for name, result in models.items()
            }).T
            if "f1_mean" in cv_comparison.columns:
                st.subheader("Training-only cross-validation")
                st.dataframe(
                    cv_comparison[["f1_mean", "f1_std", "roc_auc_mean", "pr_auc_mean"]]
                    .rename(columns={
                        "f1_mean": "Mean F1",
                        "f1_std": "F1 standard deviation",
                        "roc_auc_mean": "Mean ROC-AUC",
                        "pr_auc_mean": "Mean PR-AUC",
                    })
                    .style.format("{:.3f}"),
                    width="stretch",
                )

            selected_metrics = models.get(metrics.get("selected_model"), {})
            confusion = selected_metrics.get("confusion_matrix", {})
            if confusion:
                st.subheader("Selected model: churn confusion counts")
                cm = pd.DataFrame(
                    [
                        [confusion.get("true_negative", 0), confusion.get("false_positive", 0)],
                        [confusion.get("false_negative", 0), confusion.get("true_positive", 0)],
                    ],
                    index=["Actually stayed", "Actually churned"],
                    columns=["Predicted stayed", "Predicted churned"],
                )
                st.dataframe(cm, width="stretch")
                st.caption(
                    "False negatives are churners the model missed. False positives are customers flagged as churn risks who stayed."
                )

        feature_importance = metrics.get("feature_importance", [])
        if feature_importance:
            st.subheader("Top model signals")
            importance_frame = pd.DataFrame(feature_importance[:10]).set_index("feature")
            importance_frame.index = importance_frame.index.str.replace(
                r"^(numbers|categories)__", "", regex=True
            )
            st.bar_chart(importance_frame, y_label="Model signal magnitude")
            st.caption("Importance describes how the selected model used features; it does not establish causation or explain an individual prediction.")

        st.info(
            "Recall answers how many actual churners were found. Precision answers how many flagged customers actually churned. "
            "F1 balances those two measures; ROC-AUC and PR-AUC measure ranking quality across thresholds."
        )
        st.warning("This is a single held-out evaluation. Compare results cautiously; these metrics do not guarantee performance on future customers.")


else:
    st.title("Explore the dataset")
    st.markdown('<p class="section-note">Aggregate patterns describe this sample and should not be interpreted as causes of churn.</p>', unsafe_allow_html=True)
    if data is None:
        if data_load_error:
            st.error(f"Could not read the dataset CSV: {data_load_error}")
        else:
            st.error("Dataset CSV not found. Confirm the CSV is in the project's `data` folder.")
    else:
        st.subheader("Refine the customer sample")
        filter_columns = st.columns(4)
        filter_options = {
            "Contract": sorted(data["Contract"].dropna().unique().tolist()),
            "InternetService": sorted(data["InternetService"].dropna().unique().tolist()),
            "PaymentMethod": sorted(data["PaymentMethod"].dropna().unique().tolist()),
            "Churn": ["No", "Yes"],
        }
        selected_filters = {}
        for column, (field, options) in zip(filter_columns, filter_options.items()):
            with column:
                selected_filters[field] = st.multiselect(
                    field.replace("InternetService", "Internet service").replace("PaymentMethod", "Payment method"),
                    options,
                    default=options,
                    key=f"explore_{field}",
                )

        filtered_data = data.copy()
        for field, values in selected_filters.items():
            filtered_data = filtered_data[filtered_data[field].isin(values)]

        if filtered_data.empty:
            st.info("No customers match these filters. Adjust the selections to see results.")
        else:
            churn_rate = filtered_data["Churn"].eq("Yes").mean() * 100
            avg_monthly_charges = filtered_data["MonthlyCharges"].mean()
            avg_tenure = filtered_data["tenure"].mean()

            metric_columns = st.columns(4)
            metric_columns[0].metric("Customers in view", f"{len(filtered_data):,}")
            metric_columns[1].metric("Churn rate", f"{churn_rate:.1f}%")
            metric_columns[2].metric("Avg. monthly charges", f"${avg_monthly_charges:,.2f}")
            metric_columns[3].metric("Avg. tenure", f"{avg_tenure:.1f} months")

            st.download_button(
                "Download filtered customers",
                data=filtered_data.to_csv(index=False).encode("utf-8"),
                file_name="filtered_churn_customers.csv",
                mime="text/csv",
                width="stretch",
            )

            d1, d2 = st.columns(2)
            with d1:
                st.subheader("Churn distribution")
                counts = (
                    filtered_data["Churn"]
                    .value_counts()
                    .reindex(["No", "Yes"], fill_value=0)
                    .rename(index={"No": "Stayed", "Yes": "Churned"})
                )
                st.bar_chart(counts, y_label="Customers")
            with d2:
                st.subheader("Churn rate by contract")
                rates = (
                    filtered_data.assign(Churned=filtered_data["Churn"].eq("Yes"))
                    .groupby("Contract", observed=False)["Churned"]
                    .mean()
                    .mul(100)
                    .sort_values(ascending=False)
                )
                st.bar_chart(rates, y_label="Churn rate (%)")
                st.dataframe(rates.rename("Churn rate (%)").round(1), width="stretch")

            with st.expander("Preview filtered customers"):
                st.dataframe(filtered_data.head(100), width="stretch")
                st.caption(f"Showing up to 100 of {len(filtered_data):,} matching rows. Download the CSV for the full filtered sample.")

st.divider()
st.caption("Customer Churn Analytics and Retention Support · End-semester project · Predictions are estimates, not guarantees.")
