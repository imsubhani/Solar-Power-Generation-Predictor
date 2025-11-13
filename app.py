import streamlit as st
import numpy as np
import joblib

# Load saved model and preprocessing objects
model = joblib.load('poly_model.pkl')
poly = joblib.load('poly_features.pkl')
scaler = joblib.load('scaler.pkl')

# App title and intro
st.set_page_config(page_title="☀️ Solar Power Prediction Dashboard", layout="centered", initial_sidebar_state="expanded")
st.title("☀️ Solar Power Generation Predictor")
st.markdown("""
To predict power, the model balances two key factors: **Irradiation** (the "fuel" from sunlight) which **increases** power, and **Heat** (the "efficiency killer") which **decreases** it. The final prediction is the result of these two opposing forces.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Input Features")

# --- Step 1: The "Driver" Slider ---
irradiation = st.sidebar.slider(
    "☀️ Irradiation (kW/m²)", 
    min_value=0.0, max_value=1.2, value=0.6, step=0.05,
    help="This is the most important factor! It's the amount of sunlight hitting the panel. 0.0 means night, while 1.2 is peak sun."
)

# --- Step 2: Define Dynamic Ranges based on Irradiation ---
if irradiation == 0.0:
    st.sidebar.markdown("<p style='text-align: center; color: gray;'>Night</p>", unsafe_allow_html=True)
    # At night, temps are cool and module is near ambient
    mod_min, mod_max = 15.0, 35.0
    amb_min, amb_max = 15.0, 30.0
    mod_val, amb_val = 20.0, 20.0 

elif irradiation <= 0.3:
    st.sidebar.markdown("<p style='text-align: center; color: gray;'>Low Light / Overcast</p>", unsafe_allow_html=True)
    # Low light, module is a bit warmer than air
    mod_min, mod_max = 18.0, 49.0
    amb_min, amb_max = 20.0, 40.0
    mod_val, amb_val = 30.0, 25.0

elif irradiation <= 0.7:
    st.sidebar.markdown("<p style='text-align: center; color: gray;'>Medium Light / Sunny</p>", unsafe_allow_html=True)
    # Medium light, module is getting hot
    mod_min, mod_max = 27.0, 63.0
    amb_min, amb_max = 23.0, 39.0
    mod_val, amb_val = 50.0, 30.0

else: # High Light
    st.sidebar.markdown("<p style='text-align: center; color: gray;'>High Light / Bright Sun</p>", unsafe_allow_html=True)
    # High light, module is very hot
    mod_min, mod_max = 39.0, 70.0  # Max 70C based on analysis
    amb_min, amb_max = 25.0, 40.0  # Max 40C based on analysis
    mod_val, amb_val = 60.0, 32.0

# --- Step 3: The "Dependent" Sliders ---
module_temp = st.sidebar.slider(
    "🌡️ Module Temperature (°C)", 
    min_value=mod_min, max_value=mod_max, value=mod_val, step=0.5,
    help="The panel's surface temperature. This is a critical factor! Hotter panels are *less* efficient and produce *less* power."
)
ambient_temp = st.sidebar.slider(
    "🌤️ Ambient Temperature (°C)", 
    min_value=amb_min, max_value=amb_max, value=amb_val, step=0.5,
    help="The outside air temperature. This has a smaller effect, but still helps the model fine-tune its prediction."
)

# --- Prediction Logic ---
if st.sidebar.button("🔍 Predict Power Output"):

    features = np.array([[irradiation, module_temp, ambient_temp]])
    features_scaled = scaler.transform(features)
    features_poly = poly.transform(features_scaled)
    
    prediction_raw = model.predict(features_poly)[0]
    prediction_final = np.clip(prediction_raw, 0, None)
    
    st.success(f"### ⚡ Predicted AC Power Output: **{prediction_final:.2f} kW**")

    if prediction_final > 0:
        st.balloons()
            
    st.info(
        f"""
        **Model Scope:** This model was trained on data from a specific, hot, sunny climate. 
        The dynamic temperature ranges are based on the real data from that location.  \nThe model would not be accurate for a different climate (e.g., a cold or snowy region) 
        as it was never trained on that type of data.
        """
    )

# Footer
st.markdown("---")
st.markdown("Polynomial Regression Model (Degree 2)")