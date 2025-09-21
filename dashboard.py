
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


# ==============================
# Fairness Analysis Function
# ==============================
def run_fairness_analysis(df, target_col, protected_col):
    # Encode target column (Yes/No → 1/0)
    le = LabelEncoder()
    y_true = le.fit_transform(df[target_col])

    # Dummy prediction for now (replace with model later)
    y_pred = y_true  

    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "Selection Rate": selection_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=df[protected_col],
    )

    results = mf.by_group
    overall = mf.overall

    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=df[protected_col]
    )
    eo_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=df[protected_col]
    )

    return results, overall, dp_diff, eo_diff


# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Bias Buster Dashboard", layout="wide")
st.title("⚖️ Bias Buster Dashboard")


# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, DOCX, or PDF)", type=["csv", "docx", "pdf"]
)

# Dummy variables for testing
before_metrics = None
after_metrics = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Dataset Preview")
            st.dataframe(df.head())

            # User selects target column (label)
            target_col = st.selectbox("🎯 Select Target Column (Label)", df.columns)

            # User selects protected attribute
            protected_col = st.selectbox("🛡️ Select Protected Attribute", df.columns)

            # Run fairness analysis button
            if st.button("Run Fairness Analysis", key="fair_btn"):
                results, overall, dp_diff, eo_diff = run_fairness_analysis(
                    df, target_col, protected_col
                )

                st.subheader("📊 Overall Metrics")
                st.dataframe(overall)

                st.subheader("📊 Metrics by Group")
                st.dataframe(results)

                st.write(f"**Demographic Parity Difference:** {dp_diff:.3f}")
                st.write(f"**Equalized Odds Difference:** {eo_diff:.3f}")

                # Dummy bias analysis (replace with actual logic later)
                before_metrics = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall"],
                    "Value": [0.82, 0.78, 0.75]
                })

                after_metrics = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall"],
                    "Value": [0.87, 0.83, 0.80]
                })

                # Save dataset into session_state for bias analysis
                st.session_state["dataset"] = df
                st.session_state["before_metrics"] = before_metrics
                st.session_state["after_metrics"] = after_metrics

    except Exception as e:
        st.error(f"❌ Error while processing file: {e}")


# ------------------------------
# Run Bias Analysis (Before vs After)
# ------------------------------
if st.button("Run Bias Analysis", key="bias_btn"):
    if "before_metrics" in st.session_state and "after_metrics" in st.session_state:
        before_metrics = st.session_state["before_metrics"]
        after_metrics = st.session_state["after_metrics"]

        st.subheader("📊 Before Metrics")
        st.dataframe(before_metrics)

        st.subheader("📊 After Metrics")
        st.dataframe(after_metrics)

        # Difference
        diff = after_metrics.set_index("Metric") - before_metrics.set_index("Metric")
        diff.reset_index(inplace=True)

        st.subheader("📊 Difference (After - Before)")
        st.dataframe(diff)
        st.bar_chart(diff.set_index("Metric"))

        # ------------------------------
        # Save Results Locally
        # ------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)

        before_metrics.to_csv(f"results/before_metrics_{timestamp}.csv", index=False)
        after_metrics.to_csv(f"results/after_metrics_{timestamp}.csv", index=False)
        diff.to_csv(f"results/diff_metrics_{timestamp}.csv", index=False)

        # ------------------------------
        # Download Options
        # ------------------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            before_metrics.to_excel(writer, sheet_name="Before Metrics", index=False)
            after_metrics.to_excel(writer, sheet_name="After Metrics", index=False)
            diff.to_excel(writer, sheet_name="Difference", index=False)

        st.success(f"✅ Results saved in 'results/' folder with timestamp {timestamp}")

        # Excel download (Main page)
        st.download_button(
            label="⬇️ Download Full Results (Excel)",
            data=output.getvalue(),
            file_name=f"bias_analysis_results_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # CSV download (Main page)
        combined_csv = pd.concat(
            [
                before_metrics.assign(Type="Before"),
                after_metrics.assign(Type="After"),
                diff.assign(Type="Difference"),
            ]
        )
        st.download_button(
            label="⬇️ Download Full Results (CSV)",
            data=combined_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"bias_analysis_results_{timestamp}.csv",
            mime="text/csv",
        )

    else:
        st.error("❌ Please upload a dataset and run fairness analysis first!")
=======
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


# ==============================
# Fairness Analysis Function
# ==============================
def run_fairness_analysis(df, target_col, protected_col):
    # Encode target column (Yes/No → 1/0)
    le = LabelEncoder()
    y_true = le.fit_transform(df[target_col])

    # Dummy prediction for now (replace with model later)
    y_pred = y_true  

    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "Selection Rate": selection_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=df[protected_col],
    )

    results = mf.by_group
    overall = mf.overall

    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=df[protected_col]
    )
    eo_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=df[protected_col]
    )

    return results, overall, dp_diff, eo_diff


# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Bias Buster Dashboard", layout="wide")
st.title("⚖️ Bias Buster Dashboard")


# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, DOCX, or PDF)", type=["csv", "docx", "pdf"]
)

# Dummy variables for testing
before_metrics = None
after_metrics = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Dataset Preview")
            st.dataframe(df.head())

            # User selects target column (label)
            target_col = st.selectbox("🎯 Select Target Column (Label)", df.columns)

            # User selects protected attribute
            protected_col = st.selectbox("🛡️ Select Protected Attribute", df.columns)

            # Run fairness analysis button
            if st.button("Run Fairness Analysis", key="fair_btn"):
                results, overall, dp_diff, eo_diff = run_fairness_analysis(
                    df, target_col, protected_col
                )

                st.subheader("📊 Overall Metrics")
                st.dataframe(overall)

                st.subheader("📊 Metrics by Group")
                st.dataframe(results)

                st.write(f"**Demographic Parity Difference:** {dp_diff:.3f}")
                st.write(f"**Equalized Odds Difference:** {eo_diff:.3f}")

                # Dummy bias analysis (replace with actual logic later)
                before_metrics = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall"],
                    "Value": [0.82, 0.78, 0.75]
                })

                after_metrics = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall"],
                    "Value": [0.87, 0.83, 0.80]
                })

                # Save dataset into session_state for bias analysis
                st.session_state["dataset"] = df
                st.session_state["before_metrics"] = before_metrics
                st.session_state["after_metrics"] = after_metrics

    except Exception as e:
        st.error(f"❌ Error while processing file: {e}")


# ------------------------------
# Run Bias Analysis (Before vs After)
# ------------------------------
if st.button("Run Bias Analysis", key="bias_btn"):
    if "before_metrics" in st.session_state and "after_metrics" in st.session_state:
        before_metrics = st.session_state["before_metrics"]
        after_metrics = st.session_state["after_metrics"]

        st.subheader("📊 Before Metrics")
        st.dataframe(before_metrics)

        st.subheader("📊 After Metrics")
        st.dataframe(after_metrics)

        # Difference
        diff = after_metrics.set_index("Metric") - before_metrics.set_index("Metric")
        diff.reset_index(inplace=True)

        st.subheader("📊 Difference (After - Before)")
        st.dataframe(diff)
        st.bar_chart(diff.set_index("Metric"))

        # ------------------------------
        # Save Results Locally
        # ------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)

        before_metrics.to_csv(f"results/before_metrics_{timestamp}.csv", index=False)
        after_metrics.to_csv(f"results/after_metrics_{timestamp}.csv", index=False)
        diff.to_csv(f"results/diff_metrics_{timestamp}.csv", index=False)

        # ------------------------------
        # Download Options
        # ------------------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            before_metrics.to_excel(writer, sheet_name="Before Metrics", index=False)
            after_metrics.to_excel(writer, sheet_name="After Metrics", index=False)
            diff.to_excel(writer, sheet_name="Difference", index=False)

        st.success(f"✅ Results saved in 'results/' folder with timestamp {timestamp}")

        # Excel download (Main page)
        st.download_button(
            label="⬇️ Download Full Results (Excel)",
            data=output.getvalue(),
            file_name=f"bias_analysis_results_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # CSV download (Main page)
        combined_csv = pd.concat(
            [
                before_metrics.assign(Type="Before"),
                after_metrics.assign(Type="After"),
                diff.assign(Type="Difference"),
            ]
        )
        st.download_button(
            label="⬇️ Download Full Results (CSV)",
            data=combined_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"bias_analysis_results_{timestamp}.csv",
            mime="text/csv",
        )

    else:
        st.error("❌ Please upload a dataset and run fairness analysis first!")
>>>>>>> 27373b4 (Initial commit - Bias Buster project)
