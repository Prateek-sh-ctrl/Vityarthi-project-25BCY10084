import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="FraudGuard: Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

try:
    model = joblib.load('models/fraud_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run `src/train_model.py` first.")
    st.stop()

st.title("FraudGuard: Real-Time Transaction Analysis 🛡️")
st.write(
    "Enter transaction details below to check for potential fraud. "
    "This demo uses a pre-trained RandomForest model to predict whether a transaction is legitimate or fraudulent."
)

with st.form("transaction_form"):
    st.header("Transaction Details")

    
    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=1000.0, step=100.0)
        amount_val = st.number_input("Amount (in currency)", value=100.0, step=10.0)
    with col2:
        st.info(
            "For this demo, the 'V' features (anonymous transaction variables) are simulated. "
            "Adjust the 'Amount' to see how it affects the prediction."
        )

    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    scaled_time = scaler.transform(np.array([[time_val]]))
    scaled_amount = scaler.transform(np.array([[amount_val]]))

    v_features = np.zeros(28)

    transaction_features = np.concatenate([v_features, scaled_amount.flatten(), scaled_time.flatten()]).reshape(1, -1)

    prediction = model.predict(transaction_features)
    prediction_proba = model.predict_proba(transaction_features)

    st.header("Analysis Result")
    if prediction[0] == 1:
        st.error("🔴 Alert: This transaction is likely FRAUDULENT!", icon="🚨")
        st.write(f"Confidence Score: {prediction_proba[0][1]*100:.2f}%")
        st.warning(
            "**Action Recommended:** Please review this transaction immediately. "
            "Consider contacting the cardholder to verify."
        )
    else:
        st.success("🟢 OK: This transaction appears to be LEGITIMATE.", icon="✅")
        st.write(f"Confidence Score: {prediction_proba[0][0]*100:.2f}%")

    with st.expander("Show Technical Details"):
        st.write("Feature values sent to the model:")
        st.json({
            'scaled_amount': scaled_amount.flatten()[0],
            'scaled_time': scaled_time.flatten()[0],
            'v_features': 'zeros_for_demo'
        })
        st.write("Prediction probabilities (Class 0: Legit, Class 1: Fraud):")
        st.write(prediction_proba)

st.sidebar.header("About FraudGuard")
st.sidebar.info(
    "This application is a demonstration of a machine learning model for detecting credit card fraud. "
    "It uses a RandomForestClassifier trained on a dataset of transactions, with SMOTE to handle class imbalance."
)
st.sidebar.header("How it Works")
st.sidebar.markdown(
    """
    1.  **Input**: You provide the transaction `Time` and `Amount`.
    2.  **Preprocessing**: The inputs are scaled using the same `RobustScaler` from training.
    3.  **Prediction**: The model predicts if the transaction is fraudulent.
    4.  **Output**: A clear alert (Red for Fraud, Green for Legit) is displayed.
    """
)
