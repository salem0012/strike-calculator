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

st.write(input_df)

# Predict
if st.button("Predict Stroke Risk"):
    dmatrix = xgb.DMatrix(input_df)
    
    # Predict 1-year risk from the model
    one_year_risk = model.predict(dmatrix)[0]

    # Extrapolate to 10-year risk using compounding
    ten_year_risk = 1 - (1 - one_year_risk) ** 10

    # Show both
    st.success(f"Estimated 1-year stroke risk: {one_year_risk * 100:.2f}%")
    st.info(f"Estimated 10-year stroke risk (compounded): {ten_year_risk * 100:.2f}%")

    # Optional: add a tooltip or explanation
    with st.expander("ℹ️ What does this mean?"):
        st.markdown("""
        - The 1-year stroke risk is directly predicted by a machine learning model trained on NHANES data.
        - The 10-year risk is estimated using a compounding formula:  
          \n `1 - (1 - risk)^10`  
        - This is **not a substitute** for longitudinal prediction like the Framingham Risk Score.
        """)
{risk:.2f}%")
