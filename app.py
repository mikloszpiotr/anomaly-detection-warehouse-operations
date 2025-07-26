
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from src.features.build_features import build_features

st.set_page_config(page_title="Warehouse Anomaly Detection", layout="wide")
st.title("🏭 Anomaly Detection in Warehouse Operations")

# Sidebar
menu = st.sidebar.radio("📌 Navigation", ["Upload Data", "Anomaly Visuals", "KPI Summary", "Anomaly Table", "Export"])

# Global state
if "df" not in st.session_state:
    st.session_state.df = None

# Upload tab
if menu == "Upload Data":
    uploaded_file = st.file_uploader("📤 Upload warehouse log CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        st.session_state.df = df

        X, _ = build_features(df)
        model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
        model.fit(X)
        preds = model.predict(X)
        df["anomaly_flag"] = (preds == -1).astype(int)
        st.session_state.df = df

        st.success(f"✅ {df['anomaly_flag'].sum()} anomalies detected from {len(df)} records.")
        st.dataframe(df.head())

# Visuals tab
elif menu == "Anomaly Visuals" and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("📊 Duration vs Lag with Anomalies")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="duration_sec", y="device_lag_ms", hue="anomaly_flag", alpha=0.6, ax=ax1)
    st.pyplot(fig1)

    st.subheader("🔧 Errors by Task Type")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x="task_type", y="errors", hue="anomaly_flag", ax=ax2)
    st.pyplot(fig2)

# KPIs tab
elif menu == "KPI Summary" and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("📈 Anomaly KPIs")

    kpi1 = df["anomaly_flag"].sum()
    kpi2 = (df["anomaly_flag"].mean()) * 100
    kpi3 = df[df["anomaly_flag"] == 1]["zone_id"].value_counts().idxmax()

    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Total Anomalies", f"{kpi1}")
    col2.metric("📉 Anomaly Rate", f"{kpi2:.2f}%")
    col3.metric("🏭 Top Zone (Anomalies)", f"{kpi3}")

# Table tab
elif menu == "Anomaly Table" and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("📋 Filtered Anomaly Table")
    st.dataframe(df[df["anomaly_flag"] == 1])

# Export
elif menu == "Export" and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("📥 Download Results")
    st.download_button("Download full dataset with anomaly flag",
                       data=df.to_csv(index=False),
                       file_name="warehouse_anomalies.csv",
                       mime="text/csv")
    st.download_button("Download only anomalies",
                       data=df[df["anomaly_flag"] == 1].to_csv(index=False),
                       file_name="anomalies_only.csv",
                       mime="text/csv")
else:
    if st.session_state.df is None:
        st.info("👆 Please upload a CSV file to begin.")
