
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model and encoder (make sure you saved them after training)
model = joblib.load("fruit_model.pkl")   # your trained model file
encoder = joblib.load("label_encoder.pkl")  # encoder for fruit labels

st.set_page_config(page_title="Fruit Classification", layout="centered")

st.title("🍎🍊 Fruit Classification App")
st.write("This app predicts the type of fruit based on input features.")

# Sidebar for user input
st.sidebar.header("Enter Fruit Features")

# Example features (adjust these based on your dataset)
mass = st.sidebar.number_input("Mass (g)", min_value=1, max_value=1000, value=150)
width = st.sidebar.number_input("Width (mm)", min_value=1, max_value=200, value=70)
height = st.sidebar.number_input("Height (mm)", min_value=1, max_value=200, value=80)
color_score = st.sidebar.slider("Color Score", min_value=0.0, max_value=1.0, value=0.5)

# Prepare data for prediction
features = np.array([[mass, width, height, color_score]])

if st.sidebar.button("Predict"):
    prediction = model.predict(features)
    fruit_name = encoder.inverse_transform(prediction)[0]
    st.success(f"✅ The predicted fruit is: **{fruit_name}**")
