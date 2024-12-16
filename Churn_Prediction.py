import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and column transformer
with open('xgb_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('column_transformer.pkl', 'rb') as f:
    column_transformer = pickle.load(f)

# Fungsi untuk memprediksi churn
def predict_churn(data):
    data_transformed = column_transformer.transform(data)
    prediction = loaded_model.predict(data_transformed)
    return prediction

# Interface Streamlit
st.title("E-commerce Customer Churn Prediction")

# Input data baru dari pengguna
tenure = st.number_input("Tenure", min_value=0)
warehouse_to_home = st.number_input("WarehouseToHome")
number_of_devices = st.number_input("NumberOfDeviceRegistered")
prefered_order = st.selectbox("Preferred Order Category", ['Mobile Phone', 'laptop & accessory', 'home appliance'])
satisfaction_score = st.number_input("SatisfactionScore")
marital_status = st.selectbox("Marital Status", ['Married', 'Single'])
number_of_address = st.number_input("NumberOfAddress")
complain = st.number_input("Complain")
day_since_last_order = st.number_input("DaySinceLastOrder")
cashback_amount = st.number_input("CashbackAmount")

# Create a new data frame
new_data = pd.DataFrame({
    'Tenure': [tenure],
    'WarehouseToHome': [warehouse_to_home],
    'NumberOfDeviceRegistered': [number_of_devices],
    'PreferedOrderCat': [prefered_order],
    'SatisfactionScore': [satisfaction_score],
    'MaritalStatus': [marital_status],
    'NumberOfAddress': [number_of_address],
    'Complain': [complain],
    'DaySinceLastOrder': [day_since_last_order],
    'CashbackAmount': [cashback_amount]
})

# Make prediction
prediction = predict_churn(new_data)
st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
