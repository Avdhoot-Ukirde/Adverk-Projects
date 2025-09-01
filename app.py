
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and label encoder

model = joblib.load("fruit_model.pkl")       # trained Random Forest (or best model)
encoder = joblib.load("label_encoder.pkl")   # label encoder for fruit_name

# Streamlit App Layout

st.set_page_config(page_title="Fruit Classification", layout="centered")
st.title("🍎🍊 Fruit Classification App")
st.write("Predict the type of fruit based on its features.")

# Sidebar: User Input

st.sidebar.header("Enter Fruit Features")

mass = st.sidebar.number_input("Mass (g)", min_value=1, max_value=1000, value=150)
width = st.sidebar.number_input("Width (mm)", min_value=1, max_value=200, value=70)
height = st.sidebar.number_input("Height (mm)", min_value=1, max_value=200, value=80)
color_score = st.sidebar.slider("Color Score", min_value=0.0, max_value=1.0, value=0.5)

# Prepare features as DataFrame

feature_names = ['mass', 'width', 'height', 'color_score']   # must match training columns
features = pd.DataFrame([[mass, width, height, color_score]], columns=feature_names)

# Predict Button

if st.sidebar.button("Predict"):
    prediction = model.predict(features)
    fruit_name = encoder.inverse_transform(prediction)[0]
    st.success(f"✅ The predicted fruit is: **{fruit_name}**")
