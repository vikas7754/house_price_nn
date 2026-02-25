import streamlit as st
import torch
import sys
import os
import pandas as pd

# Add root to python path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import HousePriceNN
from core.custom_op_model import HousePriceModelWithCustomOp
from core.device import get_device

device = get_device()

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏡 House Price Prediction Neural Network")
st.write("This app uses a PyTorch Neural Network trained on housing data to predict prices.")

models = ["house_price_model_fx.pth", "house_price_model_eager.pth", "house_price_model_torchscript.pth", "house_price_model_torch_compile.pth"]

@st.cache_resource
def load_model(model_name):
    if model_name == "house_price_model_fx.pth":
        model = HousePriceModelWithCustomOp()
    else:
        model = HousePriceNN()
    model_path = os.path.join(os.path.dirname(__file__), '../models/', model_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("No trained model found, using random weights.")
    model.to(device)
    model.eval()
    return model

# Model Selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    models,
    index=0,  # Default to first model (house_price_model_fx.pth)
    help="Select which trained model to use for predictions"
)

model = load_model(selected_model)

# Input features
st.sidebar.header("House Features")
income = st.sidebar.number_input("Avg. Area Income", value=79545.45857431678)
house_age = st.sidebar.number_input("Avg. Area House Age", value=5.682861321615587)
rooms = st.sidebar.number_input("Avg. Area Number of Rooms", value=7.009188142792237)
bedrooms = st.sidebar.number_input("Avg. Area Number of Bedrooms", value=4.09)
population = st.sidebar.number_input("Area Population", value=23086.800502686456)
# Load stats
stats_path = os.path.join(os.path.dirname(__file__), '..', 'training_stats.pt')
if os.path.exists(stats_path):
    stats = torch.load(stats_path, map_location=device)
    MEAN_VALS = stats['feature_mean'].to(device)
    STD_VALS = stats['feature_std'].to(device)
    TARGET_MEAN = stats['target_mean']
    TARGET_STD = stats['target_std']
else:
    st.error("Training stats not found! Please run training first.")
    st.stop()

if st.button("Predict Price"):
    # Normalize input
    x_input = torch.tensor([income, house_age, rooms, bedrooms, population], device=device).float()
    x_input = (x_input - MEAN_VALS) / (STD_VALS + 1e-6)

    with torch.no_grad():
        prediction_norm = model(x_input.unsqueeze(0)).item()
    
    # Denormalize output
    prediction = (prediction_norm * TARGET_STD.item()) + TARGET_MEAN.item()
    
    st.success(f"Predicted Price: ${prediction:,.2f}")

    # Debug Info
    st.subheader("Debug Info")
    st.write("Model output (raw / normalized):", prediction_norm)
    st.write("Input tensor (normalized):", x_input)

    # Visualization of feature contribution (simple proxy via weights)
    st.subheader("Implementation Details")
    st.info("""
    This model uses a simple Feed-Forward Neural Network:
    - Input Layer: 5 Features
    - Hidden Layer 1: 64 Neurons + ReLU
    - Hidden Layer 2: 64 Neurons + ReLU
    - Output Layer: 1 Neuron (Price)
    """)

# Runtime Comparison Section
st.markdown("---")
st.header(f"⚡ Runtime Comparison ({device.type.upper()})")
st.caption("Lower runtime is better. Measurements were taken offline using the same data and batch size.")

metrics_path = os.path.join(os.path.dirname(__file__), 'runtime_metrics.json')

if os.path.exists(metrics_path):
    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create DataFrame for display
    df_metrics = pd.DataFrame(list(metrics.items()), columns=['Execution Mode', 'Avg Time (ms)'])
    df_metrics = df_metrics.sort_values(by='Avg Time (ms)')

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Average Training Step Time:")
        st.dataframe(df_metrics, hide_index=True)
    
    with col2:
        st.bar_chart(df_metrics.set_index('Execution Mode'))

else:
    st.warning("⚠️ No runtime metrics found. Run `python benchmark_runtime.py` to generate them.")
