import streamlit as st
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("rf_classifier_model.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------- CENTER ALIGN CSS ----------------
st.markdown("""
    <style>
    .main {
        display: flex;
        justify-content: center;
    }
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details below:")

# ---------------- INPUTS ----------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)

# Encode sex
sex_encoded = le.transform([sex])[0]

# ---------------- PREDICT ----------------
if st.button("Predict"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did Not Survive")