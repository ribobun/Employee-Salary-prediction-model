import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("svm_salary_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

st.title("Employee Salary Estimation App")

# Input fields
department = st.selectbox("Department", ['Quality Control','Manufacturing', 'Product Development', 'Sales'])
years = st.slider("Years of Experience", 0, 40, 5)
job_rate = st.slider("Job Performance Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
sick_leaves = st.number_input("Number of Sick Leaves", min_value=0, max_value=60, value=5)
unpaid_leaves = st.number_input("Number of Unpaid Leaves", min_value=0, max_value=60, value=3)

# Convert department
encoded_dept = label_encoder.transform([department])[0]

# Predict
if st.button("Estimate Salary"):
    features = np.array([[encoded_dept, years, job_rate, sick_leaves, unpaid_leaves]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Annual Salary: {prediction:,.2f}")
