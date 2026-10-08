import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# --- Load the trained model ---
# Use caching so the model is loaded only once
@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- App Title and Description ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application predicts the likelihood of heart disease based on clinical parameters.
It uses a **Random Forest** classifier trained on the Heart Disease dataset.
""")

st.divider()

# --- Input Form ---
st.subheader("Enter Patient Details")

# Use columns for a neat layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                      format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 
                                              2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                       format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2], 
                           format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality", 
                                                   2: "Left ventricular hypertrophy"}[x])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                         format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], 
                         format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], 
                        format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 
                                                2: "Reversible Defect", 3: "Unknown"}[x])

# --- Prediction ---
if st.button("Predict Heart Disease Risk", type="primary"):
    # Create a DataFrame with the input data in the correct order
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
        "oldpeak", "slope", "ca", "thal"
    ])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.divider()
    
    # Display result
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**")
        st.write(f"Confidence: {prediction_proba[1]:.2%}")
    else:
        st.success(f"✅ **Low Risk of Heart Disease**")
        st.write(f"Confidence: {prediction_proba[0]:.2%}")
    
    st.caption("Disclaimer: This is a demo application and not a medical diagnosis tool.")