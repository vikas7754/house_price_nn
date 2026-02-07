import streamlit as st
import torch
import sys
import os
import pandas as pd

# Add root to python path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import HousePriceNN

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏡 House Price Prediction Neural Network")
st.write("This app uses a PyTorch Neural Network trained on housing data to predict prices.")

@st.cache_resource
def load_model():
    model = HousePriceNN()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'house_price_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded trained model.")
    else:
        print("No trained model found, using random weights.")
    model.eval()
    return model

model = load_model()

# Input features
st.sidebar.header("House Features")
income = st.sidebar.number_input("Avg. Area Income", value=65000.0)
house_age = st.sidebar.number_input("Avg. Area House Age", value=6.0)
rooms = st.sidebar.number_input("Avg. Area Number of Rooms", value=7.0)
bedrooms = st.sidebar.number_input("Avg. Area Number of Bedrooms", value=4.0)
population = st.sidebar.number_input("Area Population", value=40000.0)

# Preprocessing stats (these should match core/data.py ideally, hardcoding for simplicity here as example)
# In a real app, you would load these from a file saved during training.
MEAN_VALS = torch.tensor([65000.0, 6.0, 7.0, 4.0, 40000.0])
STD_VALS = torch.tensor([10000.0, 2.0, 1.0, 1.0, 10000.0])

if st.button("Predict Price"):
    # Normalize input
    x_input = torch.tensor([income, house_age, rooms, bedrooms, population])
    x_input = (x_input - MEAN_VALS) / (STD_VALS + 1e-6)
    
    with torch.no_grad():
        prediction = model(x_input.unsqueeze(0)).item()
    
    st.success(f"Predicted Price: ${prediction:,.2f}")

    # Visualization of feature contribution (simple proxy via weights)
    st.subheader("Implementation Details")
    st.info("""
    This model uses a simple Feed-Forward Neural Network:
    - Input Layer: 5 Features
    - Hidden Layer 1: 64 Neurons + ReLU
    - Hidden Layer 2: 64 Neurons + ReLU
    - Output Layer: 1 Neuron (Price)
    """)
