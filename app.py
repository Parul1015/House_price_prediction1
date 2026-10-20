import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------
# 1. Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Advanced House Price Prediction App")
st.markdown("This app uses a machine learning regression model trained on the Ames Housing dataset to predict property prices in real time.")

# ----------------------------------
# 2. Load the Trained Pipeline and Columns
# ----------------------------------
@st.cache_resource
def load_assets():
    # Loads the full pipeline (imputer + scaler + encoder + gradient boosting model)
    model = joblib.load("house_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

st.divider()

# ----------------------------------
# 3. User Input Controls for Main Features
# ----------------------------------
st.subheader("Adjust Property Features")

col1, col2, col3 = st.columns(3)

with col1:
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1500, step=50)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800, step=50)

with col2:
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=500, step=50)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2005)

with col3:
    overall_qual = st.slider("Overall Material & Finish Quality (1-10)", min_value=1, max_value=10, value=6)
    overall_cond = st.slider("Overall Condition Rating (1-10)", min_value=1, max_value=10, value=5)

st.divider()

# ----------------------------------
# 4. Prediction Execution
# ----------------------------------
if st.button("Predict Property Price", type="primary"):
    # Create a blank dataframe matching all columns the model was trained on
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Fill in the key user inputs matching your notebook's feature names
    if 'GrLivArea' in input_data.columns:
        input_data['GrLivArea'] = gr_liv_area
    if 'TotalBsmtSF' in input_data.columns:
        input_data['TotalBsmtSF'] = total_bsmt_sf
    if 'GarageArea' in input_data.columns:
        input_data['GarageArea'] = garage_area
    if 'YearBuilt' in input_data.columns:
        input_data['YearBuilt'] = year_built
    if 'OverallQual' in input_data.columns:
        input_data['OverallQual'] = overall_qual
    if 'OverallCond' in input_data.columns:
        input_data['OverallCond'] = overall_cond

    # Make prediction (returns log-transformed price because model was trained on y_log)
    prediction_log = model.predict(input_data)
    
    # Convert log-transformed scale back to actual dollar amounts using expm1
    try:
        final_price = np.expm1(prediction_log[0])
    except:
        final_price = prediction_log[0]

    # Display real-time result
    st.success(f"### Estimated House Sale Price: ${final_price:,.2f}")
    st.balloons()
