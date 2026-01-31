import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="centered"
)

# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("housing_price_model.joblib")

model = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🏡 California Housing Price Predictor")
st.write("Enter house characteristics to estimate the median house value.")

st.divider()
st.subheader("📊 House Features")

# --------------------------------------------------
# Inputs (bounded using dataset statistics)
# --------------------------------------------------
longitude = st.slider(
    "Longitude",
    min_value=-124.35,
    max_value=-114.31,
    value=-118.49,
    step=0.01
)

latitude = st.slider(
    "Latitude",
    min_value=32.54,
    max_value=41.95,
    value=34.26,
    step=0.01
)

housing_median_age = st.slider(
    "Housing Median Age (years)",
    min_value=1,
    max_value=52,
    value=29
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1,
    max_value=40000,
    value=2127,
    step=10
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1,
    max_value=7000,
    value=435,
    step=5
)

population = st.number_input(
    "Population",
    min_value=1,
    max_value=40000,
    value=1166,
    step=10
)

households = st.number_input(
    "Households",
    min_value=1,
    max_value=7000,
    value=409,
    step=5
)

median_income = st.slider(
    "Median Income (× $10,000)",
    min_value=0.5,
    max_value=15.0,
    value=3.53,
    step=0.1
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# --------------------------------------------------
# Validation
# --------------------------------------------------
if total_bedrooms > total_rooms:
    st.warning("⚠️ Total bedrooms cannot exceed total rooms.")
    st.stop()

# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button("💰 Predict House Value"):
    prediction = model.predict(input_data)[0]

    st.success(
        f"🏠 Estimated Median House Value: **${prediction:,.0f}**"
    )

    st.caption("Prediction generated using a Random Forest model trained on California housing data.")

