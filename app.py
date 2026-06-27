import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# --- 1. SET PAGE CONFIG & TITLE ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🏥 Diabetes Prediction Assistant")
st.markdown("Enter the patient's details below to predict the likelihood of diabetes using Machine Learning and Deep Learning.")

# --- 2. LOAD MODELS (with Caching for Speed) ---
@st.cache_resource
def load_models():
    # Load Random Forest (Pipeline contains Preprocessor + Model)
    rf_pipeline = joblib.load('random_forest_model.pkl')
    
    # Extract the preprocessor from RF pipeline to use for ANN as well
    preprocessor = rf_pipeline.named_steps['preprocessor']
    
    # Reconstruct ANN Structure to load weights
    ann_structure = nn.Sequential(
        nn.Linear(11, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(16, 1), nn.Sigmoid()
    )
    # Load weights onto CPU safely
    ann_structure.load_state_dict(torch.load('diabetes_ann_model.pt', map_location=torch.device('cpu')))
    ann_structure.eval()
    
    return rf_pipeline, preprocessor, ann_structure

try:
    rf_model, preprocessor, ann_model = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models. Please ensure 'random_forest_model.pkl' and 'diabetes_ann_model.pt' are in the same folder. Details: {e}")

# --- 3. USER INPUT INTERFACE ---
st.header("Patient Demographics & Medical History")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=30)
    hypertension = st.selectbox("Hypertension (High Blood Pressure)", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease History", ["No", "Yes"])

with col2:
    smoking_history = st.selectbox("Smoking History", ["non smoker", "past smoker", "current"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=5.5, step=0.1)
    blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# --- 4. MODEL SELECTION ---
st.markdown("---")
selected_model = st.radio("🔮 Choose Prediction Model:", ("Random Forest (Slightly higher Precision)", "Neural Network (ANN - Best Recall/Sensitivity)"))

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Diabetes Status", type="primary"):
    # Convert input to DataFrame exactly like training format
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': float(age),
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': int(blood_glucose)
    }])
    
    if "Random Forest" in selected_model:
        # RF handles transformation inside its pipeline automatically
        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0][1]
    else:
        # ANN needs manual transformation via extracted preprocessor
        transformed_data = preprocessor.transform(input_data)
        input_tensor = torch.tensor(transformed_data, dtype=torch.float32)
        
        with torch.no_grad():
            prob_tensor = ann_model(input_tensor)
            probability = prob_tensor.item()
            prediction = 1 if probability > 0.5 else 0

    # Display Results Comfortably
    st.markdown("### 📊 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes Detected!** (Probability: {probability*100:.2f}%)")
        st.markdown("The model suggests consulting a healthcare specialist for formal clinical evaluation.")
    else:
        st.success(f"✅ **Low Risk / Normal** (Probability of Diabetes: {probability*100:.2f}%)")
        st.markdown("Patient parameters fall within standard healthy ranges based on current model thresholds.")
