import streamlit as st
import pandas as pd
import joblib

# Load the trained model from a .joblib file
@st.cache_resource
def load_model():
    model = joblib.load("pv_model.joblib")
    return model

model = load_model()

# --- UI --- #
st.title("⚡ Predict PV Output")
st.markdown("Enter the weather parameters below:")

# 🔒 Hide GitHub icon, menu, and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# All 9 features used during training
precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0)
cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=20.0)
solar_radiation = st.number_input("Solar Radiation (W/m²)", min_value=0.0, max_value=1200.0, value=800.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.25)
uv_index = st.number_input("UV Index", min_value=0.0, max_value=15.0, value=5.0)
temp_min = st.number_input("Temperature Min (°C)", min_value=-30.0, max_value=50.0, value=15.0)
temp_max = st.number_input("Temperature Max (°C)", min_value=-10.0, max_value=60.0, value=28.0)
solar_energy = st.number_input("Solar Energy (kWh)", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "precip": [precip],
        "cloud cover": [cloud_cover],
        "solar radiation": [solar_radiation],
        "humidity": [humidity],
        "pressure": [pressure],
        "UV index": [uv_index],
        "temp min": [temp_min],
        "temp max": [temp_max],
        "solar energy": [solar_energy]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted PV Output: {prediction:.2f} kW")

st.caption("Model: Random Forest Regressor | Trained on 9 features including solar energy")

# Adding the signature
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size: 12px; color: gray;">
        Created by <strong>Amanpreet Singh</strong>
    </div>
    """,
    unsafe_allow_html=True
)
