#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

# --- Load the Saved Model and Columns ---
# This is where the app loads the "brain" you trained in the other script.
try:
    model = joblib.load('risk_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run the 'train_model.py' script first to generate them.")
    st.stop()


# --- App Title and Description ---
st.title("Live Loan Risk Prediction Dashboard 📊")
st.write(
    "This dashboard predicts the risk of a new customer defaulting on their loan. "
    "Enter the customer's details in the sidebar to get a live prediction."
)


# --- Input Fields in the Sidebar ---
st.sidebar.header("Enter New Customer Details")

# Helper function to create the input fields
def user_input_features():
    age = st.sidebar.slider('Age', 21, 70, 35)
    income = st.sidebar.number_input('Annual Income (INR)', min_value=100000, max_value=5000000, value=500000, step=10000)
    employment_status = st.sidebar.selectbox('Employment Status', ('Salaried', 'Self-employed', 'Unemployed','Student'))
    loan_amount = st.sidebar.number_input('Loan Amount (INR)', min_value=10000, max_value=1000000, value=100000, step=5000)
    tenure_months = st.sidebar.slider('Loan Tenure (Months)', 6, 60, 24)
    interest_rate = st.sidebar.slider('Interest Rate (%)', 5.0, 25.0, 12.0, 0.1)

    data = {
        'Age': age,
        'Income': income,
        'EmploymentStatus': employment_status,
        'LoanAmount': loan_amount,
        'TenureMonths': tenure_months,
        'InterestRate': interest_rate
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader("Customer Details Entered:")
st.write(input_df)

# --- Prediction Logic ---
if st.sidebar.button('Calculate Risk'):
    st.subheader("Prediction:")

    # --- Feature Engineering (Done inside the dashboard) ---
    st.write("1. Performing Feature Engineering...")
    # Create the same features the model was trained on
    work_df = input_df.copy()
    work_df['DTI_Ratio'] = work_df['LoanAmount'] / work_df['Income']
    work_df['Loan_Burden_Index'] = (work_df['LoanAmount'] * work_df['InterestRate']) / work_df['TenureMonths']
    st.write(" - Debt-to-Income (DTI) Ratio Calculated.")
    st.write(" - Loan Burden Index Calculated.")


    # --- Data Preparation ---
    st.write("2. Preparing Data for Model...")
    # One-hot encode the categorical feature
    encoded_df = pd.get_dummies(work_df, columns=['EmploymentStatus'], drop_first=True)

    # Align columns with the model's training columns
    # This is a crucial step to ensure the data has the exact same structure
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, encoded_df])
    final_df = final_df.fillna(0) # Fill any missing columns (from encoding) with 0
    final_df = final_df[model_columns] # Ensure column order is the same
    st.write(" - Data structure aligned with model expectations.")


    # --- Make Prediction ---
    st.write("3. Making Prediction...")
    prediction_proba = model.predict_proba(final_df)

    # The probability of the "1" class (default)
    risk_probability = prediction_proba[0][1]

    # --- Display Results ---
    st.subheader("Results")
    st.write(f"The calculated probability of default is: **{risk_probability:.2%}**")

    # Add a visual gauge/indicator
    if risk_probability < 0.3:
        st.success("Risk Level: Low")
    elif risk_probability < 0.7:
        st.warning("Risk Level: Medium")
    else:
        st.error("Risk Level: High")



# In[ ]:




