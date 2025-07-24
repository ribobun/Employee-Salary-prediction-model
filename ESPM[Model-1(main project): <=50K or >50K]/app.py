import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("svm_salary_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.title("Employee Income Prediction App (SVM)")

# User inputs
age = st.slider("Age", 18, 70, 30)
education = st.selectbox("Education", label_encoders['education'].classes_)
marital = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
race = st.selectbox("Race", label_encoders['race'].classes_)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
hours = st.slider("Hours per Week", 1, 100, 40)
cap_gain = st.number_input("Capital Gain", value=0)
cap_loss = st.number_input("Capital Loss", value=0)
edu_num = st.slider("Education Num", 1, 16, 10)
country = st.selectbox("Native Country", label_encoders['native-country'].classes_)
workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)

# Prediction
if st.button("Predict Income"):
    # Prepare input data
    input_data = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'education': label_encoders['education'].transform([education])[0],
        'educational-num': edu_num,
        'marital-status': label_encoders['marital-status'].transform([marital])[0],
        'occupation': label_encoders['occupation'].transform([occupation])[0],
        'relationship': label_encoders['relationship'].transform([relationship])[0],
        'race': label_encoders['race'].transform([race])[0],
        'gender': label_encoders['gender'].transform([gender])[0],
        'capital-gain': cap_gain,
        'capital-loss': cap_loss,
        'hours-per-week': hours,
        'native-country': label_encoders['native-country'].transform([country])[0]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    result = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"Predicted Income: {result}")
    st.info(f"Prediction Confidence: {probability:.2%}")
