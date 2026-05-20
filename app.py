import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.title("🏭 Predictive Maintenance System")

# Load data
df = pd.read_csv("Thales_Group_Manufacturing.csv")

# Datetime
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Timestamp"],
    errors="coerce",
    dayfirst=True
)

# Features
features = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW"
]

X = df[features]

# Model
model = IsolationForest(contamination=0.05, random_state=42)

df["anomaly"] = model.fit_predict(X)

# Risk
df["risk_level"] = df["anomaly"].apply(
    lambda x: "High" if x == -1 else "Low"
)

# UI
machine = st.selectbox(
    "Select Machine",
    df["Machine_ID"].unique()
)

filtered = df[df["Machine_ID"] == machine]

st.subheader("Machine Data")
st.dataframe(filtered)

st.subheader("Temperature Trend")
st.line_chart(filtered["Temperature_C"])

st.subheader("Vibration Trend")
st.line_chart(filtered["Vibration_Hz"])