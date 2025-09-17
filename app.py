import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from PIL import Image

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

# Load feature columns
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Set page title and icon
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("Heart Disease Prediction App")
st.markdown("""
This app predicts the likelihood of a patient having heart disease based on clinical features.
Please fill in the patient information below and click **Predict** to see the results.
""")

# Sidebar with information
st.sidebar.header("About")
st.sidebar.info(
    """
    This machine learning model is trained on the Cleveland Heart Disease dataset 
    from the UCI Machine Learning Repository. It uses a Logistic Regression algorithm 
    with optimized hyperparameters to predict heart disease presence.
    
    The model achieves approximately 88.5% accuracy on test data.
    """
)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Demographics")
    age = st.slider("Age", 29, 77, 50)
    sex = st.radio("Sex", options=["Female", "Male"])
    
with col2:
    st.header("Clinical Measurements")
    cp = st.selectbox("Chest Pain Type", 
                     options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 120, 570, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    
# Additional inputs in expandable section
with st.expander("Additional Clinical Parameters"):
    col3, col4 = st.columns(2)
    
    with col3:
        restecg = st.selectbox("Resting Electrocardiographic Results", 
                              options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
        exang = st.radio("Exercise Induced Angina", options=["No", "Yes"])
        
    with col4:
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.5, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           options=["Upsloping", "Flat", "Downsloping"])
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
        thal = st.selectbox("Thalassemia", 
                          options=["Normal", "Fixed defect", "Reversible defect"])

# Convert categorical inputs to numerical values
sex_num = 1 if sex == "Male" else 0

cp_mapping = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
cp_num = cp_mapping[cp]

fbs_num = 1 if fbs == "Yes" else 0

restecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
restecg_num = restecg_mapping[restecg]

exang_num = 1 if exang == "Yes" else 0

slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope_num = slope_mapping[slope]

thal_mapping = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}
thal_num = thal_mapping[thal]

# Create feature array in the correct order
features = np.array([[
    age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, 
    thalach, exang_num, oldpeak, slope_num, ca, thal_num
]])

# Prediction button
if st.button("Predict Heart Disease"):
    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    # Display results
    st.subheader("Prediction Results")
    
    if prediction[0] == 1:
        st.error(f"The model predicts **heart disease is present** with {prediction_proba[0][1]*100:.2f}% confidence.")
        st.warning("Please consult with a healthcare professional for proper diagnosis and treatment.")
    else:
        st.success(f"The model predicts **no heart disease** with {prediction_proba[0][0]*100:.2f}% confidence.")
        st.info("Maintain a healthy lifestyle with regular exercise and balanced diet to keep your heart healthy.")
    
    # Show probability distribution
    st.write("Probability Distribution:")
    prob_df = pd.DataFrame({
        'Condition': ['No Heart Disease', 'Heart Disease Present'],
        'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
    })
    st.bar_chart(prob_df.set_index('Condition'))

# Add some information about the features
with st.expander("Feature Information"):
    st.markdown("""
    **Feature Descriptions:**
    - **Age**: Age in years
    - **Sex**: Gender (0 = female, 1 = male)
    - **Chest Pain Type**: 
        - 0 = Typical angina
        - 1 = Atypical angina
        - 2 = Non-anginal pain
        - 3 = Asymptomatic
    - **Resting Blood Pressure**: In mm Hg on admission to hospital
    - **Serum Cholesterol**: In mg/dl
    - **Fasting Blood Sugar**: > 120 mg/dl (1 = true; 0 = false)
    - **Resting ECG Results**:
        - 0 = Normal
        - 1 = ST-T wave abnormality
        - 2 = Left ventricular hypertrophy
    - **Max Heart Rate Achieved**: Maximum heart rate achieved during exercise
    - **Exercise Induced Angina**: (1 = yes; 0 = no)
    - **ST Depression**: Induced by exercise relative to rest
    - **Slope**: Slope of the peak exercise ST segment
        - 0 = Upsloping
        - 1 = Flat
        - 2 = Downsloping
    - **Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
    - **Thalassemia**: 
        - 1 = Normal
        - 2 = Fixed defect
        - 3 = Reversible defect
    """)

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.")