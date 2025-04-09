import streamlit as st
import pandas as pd
import xgboost as xgb

# Load the trained model
model = xgb.Booster()
model.load_model("strike_model.model")

st.title("STRIKE Stroke Risk Calculator")

# UI Inputs
age = st.slider("Age", 18, 100, 60)
sex = st.selectbox("Sex", ["Female", "Male"])
sbp = st.slider("Systolic Blood Pressure (SBP)", 80, 220, 120)
dbp = st.slider("Diastolic Blood Pressure (DBP)", 40, 140, 80)
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

# Encoding
sex_num = 1 if sex == "Male" else 0
diabetes_num = 1 if diabetes == "Yes" else 0
smoking_map = {"Never": 0, "Former": 1, "Current": 2}
smoking_num = smoking_map[smoking]

# Final input with only the 6 model features
input_df = pd.DataFrame([{
    "age": age,
    "dbp": dbp,
    "sbp": sbp,
    "diabetes": diabetes_num,
    "sex": sex_num,
    "smoking": smoking_num
}])

# Predict
if st.button("Predict Stroke Risk"):
    dmatrix = xgb.DMatrix(input_df)
    risk = model.predict(dmatrix)[0] * 100
    st.success(f"Predicted stroke risk: {risk:.2f}%")
