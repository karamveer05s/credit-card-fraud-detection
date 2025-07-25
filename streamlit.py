import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load components ===
@st.cache_data
def load_model_artifacts():
    return {
        "models": {
            "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
            "Random Forest": joblib.load("models/random_forest.pkl"),
            "XGBoost": joblib.load("models/xgboost.pkl")
        },
        "scaler": joblib.load("models/scaler.pkl"),
        "features": joblib.load("models/feature_columns.pkl")
    }

@st.cache_data
def load_dataset():
    return pd.read_csv("../data/fraud_dataset.csv")

# === Load data & components ===
artifacts = load_model_artifacts()
df = load_dataset()

models = artifacts["models"]
scaler = artifacts["scaler"]
features = artifacts["features"]

# === UI Title ===
st.title("💳 Credit Card Fraud Detection")

# === Sidebar inputs ===
st.sidebar.header("📋 Transaction Input")

merchant_list = sorted(df['merchant'].dropna().unique())
category_list = sorted(df['category'].dropna().unique())
job_list = sorted(df['job'].dropna().unique())

# Input Fields
merchant_input = st.sidebar.selectbox("🛒 Merchant", merchant_list, key="merchant")
category_input = st.sidebar.selectbox("📦 Category", category_list, key="category")
job_input = st.sidebar.selectbox("👨‍💼 Customer Job", job_list, key="job")
amount_input = st.sidebar.slider("💰 Transaction Amount ($)", 0.0, 5000.0, 100.0, step=10.0, key="amount")
selected_model = st.sidebar.selectbox("⚙️ Select Model", list(models.keys()), key="model")

# === Prediction Trigger ===
if "predict" not in st.session_state:
    st.session_state["predict"] = False

# Run prediction on button click
if st.sidebar.button("🚨 Detect Fraud"):
    st.session_state["predict"] = True

# === Prediction Section (only if triggered) ===
if st.session_state["predict"]:
    try:
        # Encode input values
        merchant_code = pd.Series([merchant_input], dtype="category").cat.set_categories(merchant_list).cat.codes[0]
        category_code = pd.Series([category_input], dtype="category").cat.set_categories(category_list).cat.codes[0]
        job_code = pd.Series([job_input], dtype="category").cat.set_categories(job_list).cat.codes[0]

        scaled_amt = scaler.transform([[amount_input]])[0][0]

        input_data = {
            "merchant": merchant_code,
            "category": category_code,
            "job": job_code,
            "scaled_amt": scaled_amt
        }

        for col in features:
            if col not in input_data:
                input_data[col] = 0

        input_df = pd.DataFrame([input_data])[features]

        model = models[selected_model]
        fraud_proba = model.predict_proba(input_df)[0][1]

        # === Show Result ===
        st.subheader("🔎 Prediction Result")
        st.success(f"Model Used: **{selected_model}**")
        st.metric("Fraud Probability", f"{fraud_proba * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

    # Reset trigger (optional)
    st.session_state["predict"] = False
