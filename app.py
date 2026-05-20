import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Page Config
st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="wide"
)

# Title
st.title("🏭 Predictive Maintenance and Anomaly Detection System")

# Load Dataset
df = pd.read_csv("Thales_Group_Manufacturing.csv")

# Datetime Conversion
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Timestamp"],
    errors="coerce",
    dayfirst=True
)

# Remove missing values
df = df.dropna()

# Feature Selection
features = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Error_Rate_%"
]

X = df[features]

# Isolation Forest Model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["anomaly"] = model.fit_predict(X)

# Risk Mapping
df["risk_level"] = df["anomaly"].apply(
    lambda x: "High" if x == -1 else "Low"
)

# KPI Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Machines",
        df["Machine_ID"].nunique()
    )

with col2:
    st.metric(
        "High Risk Records",
        (df["risk_level"] == "High").sum()
    )

with col3:
    st.metric(
        "Total Records",
        len(df)
    )

# Risk Distribution
st.subheader("📊 Risk Distribution")

risk_counts = df["risk_level"].value_counts()

st.bar_chart(risk_counts)

# Sidebar Filters
st.sidebar.header("Filters")

machine = st.sidebar.selectbox(
    "Select Machine",
    df["Machine_ID"].unique()
)

operation_mode = st.sidebar.selectbox(
    "Operation Mode",
    df["Operation_Mode"].unique()
)

# Filtered Data
filtered = df[
    (df["Machine_ID"] == machine) &
    (df["Operation_Mode"] == operation_mode)
]

# Charts
chart_data = filtered.set_index("datetime")

st.subheader("🌡 Temperature Trend")
st.line_chart(chart_data["Temperature_C"])

st.subheader("⚙ Vibration Trend")
st.line_chart(chart_data["Vibration_Hz"])

st.subheader("⚡ Power Consumption Trend")
st.line_chart(chart_data["Power_Consumption_kW"])

# Risk Table
st.subheader("🚨 Machine Risk Data")

st.dataframe(
    filtered[
        [
            "datetime",
            "Machine_ID",
            "Operation_Mode",
            "Temperature_C",
            "Vibration_Hz",
            "Power_Consumption_kW",
            "risk_level"
        ]
    ]
)

# High Risk Alerts
st.subheader("🔴 High Risk Alerts")

high_risk = filtered[
    filtered["risk_level"] == "High"
]

if len(high_risk) > 0:
    st.warning(
        f"{len(high_risk)} high-risk records detected!"
    )

    st.dataframe(
        high_risk[
            [
                "datetime",
                "Temperature_C",
                "Vibration_Hz",
                "Error_Rate_%"
            ]
        ]
    )

else:
    st.success("No high-risk anomalies detected.")