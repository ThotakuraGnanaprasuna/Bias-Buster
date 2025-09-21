import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Page setup ----------------
st.set_page_config(page_title="Bias Buster", layout="wide")
st.title("⚖️ Bias Buster Dashboard")
st.caption("Detect and analyze bias in datasets and models")

# ---------------- Sidebar: Data ----------------
st.sidebar.header("📂 Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
use_demo = st.sidebar.button("Use demo HR data")

# Load dataframe
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_demo:
    demo_path = Path(__file__).parent / "data" / "hr_demo.csv"
    df = pd.read_csv(demo_path)

if df is None:
    st.info("⬅️ Upload a CSV from the sidebar or click **Use demo HR data**.")
    st.stop()

# Basic cleaning
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# ---------------- Dataset Preview ----------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------- Column Selectors ----------------
st.subheader("⚙️ Configure Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    protected = st.selectbox(
        "Protected Attribute (group by)",
        options=[c for c in df.columns if df[c].nunique() <= 100],
        index=0
    )
with col2:
    outcome = st.selectbox("Outcome / Target column", options=df.columns, index=min(1, len(df.columns)-1))
with col3:
    labels = sorted(df[outcome].dropna().astype(str).unique().tolist())
    pos_label = st.selectbox("Positive label in outcome", options=labels, index=0)

st.write(f"👉 You selected: **{protected}** vs **{outcome}** | Positive label: **{pos_label}**")

# ---------------- Helper: selection rates ----------------
def selection_rates(data: pd.DataFrame, group_col: str, y_col: str, positive: str) -> pd.Series:
    y = (data[y_col].astype(str) == str(positive))
    return data.assign(_y=y).groupby(group_col)["_y"].mean().sort_index()

# ---------------- Bias Metrics ----------------
st.subheader("📊 Bias Metrics")

rates = selection_rates(df, protected, outcome, pos_label)
dpd = float(rates.max() - rates.min())                     # Demographic Parity Difference
di = float(rates.min() / rates.max()) if rates.max() > 0 else np.nan  # Disparate Impact

m1, m2, m3 = st.columns(3)
m1.metric("Groups", f"{len(rates)}")
m2.metric("Demographic Parity Difference", f"{dpd:.3f}")
m3.metric("Disparate Impact", f"{di:.3f}" if not np.isnan(di) else "NA")

if not np.isnan(di) and di < 0.80:
    st.error("⚠️ Potential bias detected (Disparate Impact < 0.80)")
else:
    st.success("✅ No strong DI signal")

# ---------------- Selection Rate by Group (bar chart) ----------------
st.subheader("📈 Selection Rate by Group")
fig, ax = plt.subplots()
rates.plot(kind="bar", ax=ax, edgecolor="black")
ax.set_xticklabels(rates.index, rotation=0)
ax.set_ylabel("Selection Rate")
ax.set_xlabel(protected)
ax.set_title(f"Selection Rate by {protected}")
st.pyplot(fig)

# ---------------- Group Metrics Table ----------------
st.subheader("📋 Group Metrics Table")

# Robust way to get per-group distribution for each outcome label
counts = df.groupby([protected, outcome]).size()
proportions = counts.groupby(level=0).apply(lambda s: s / s.sum()).unstack(fill_value=0)

group_counts = df[protected].value_counts().sort_index()
metrics_df = pd.DataFrame({"Total Count": group_counts})
metrics_df[f"Positive ({pos_label}) Rate"] = proportions.get(pos_label, 0)

# If there is at least one other label, show it as "Negative"
other_labels = [lbl for lbl in proportions.columns if str(lbl) != str(pos_label)]
if other_labels:
    neg_label = other_labels[0]
    metrics_df[f"Negative ({neg_label}) Rate"] = proportions.get(neg_label, 0)

st.dataframe(metrics_df, use_container_width=True)
