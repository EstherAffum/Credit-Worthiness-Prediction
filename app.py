 
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# 1 Good (Lower Risk), 0 Bad(Higher Risk)

folder = "C:/Users/Hi/Desktop/DESKTOP/BOOT CAMPS/Tech4Dev - Data Science/CapStone/"
model = joblib.load(folder + "random_forest_credit_model.pkl")
encoders = {col: joblib.load(folder + f"{col}_encoder.pkl") for col in {"Sex", "Housing", "Saving accounts", "Checking account"}}


st.title("Credit Risk Prediction App")
st.markdown("Please enter the applicants details to determine the credit risk.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["No_Account", "little", "moderate", "rich", "quite_rich"])
checking_account = st.selectbox("Checking Account", ["No_Account", "little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (in months)", min_value=1, value=12)

input_german = pd.DataFrame([{
    "Age": age,
    "Sex": encoders["Sex"].transform([sex])[0],
    "Job": job,
    "Housing": encoders["Housing"].transform([housing])[0],
    "Saving accounts": encoders["Saving accounts"].transform([saving_accounts])[0],
    "Checking account": encoders["Checking account"].transform([checking_account])[0],
    "Credit amount": credit_amount,
    "Duration": duration
}])

if st.button("Predict Credit Risk"):
    pred = model.predict(input_german)[0]

    if pred == 1:
        st.success("The applicant is predicted to be a Good Credit Risk (Lower Risk).")
    else:
        st.error("The applicant is predicted to be a Bad Credit Risk (Higher Risk).")


