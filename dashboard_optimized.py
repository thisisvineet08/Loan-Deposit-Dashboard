import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Model Loading ---
# This function is cached, so it only runs once, making the app fast.
@st.cache_resource
def load_models():
    """
    Loads all the saved model files and the training columns from disk.
    """
    models_path = 'models'
    if not os.path.exists(models_path):
        st.error("`models` directory not found. Please run the `01_save_models_and_columns.py` script first.")
        return None, None

    # Load the column layout the models were trained on
    try:
        train_cols = joblib.load(os.path.join(models_path, 'training_columns.joblib'))
    except FileNotFoundError:
        st.error("`training_columns.joblib` not found. Please run the saving script.")
        return None, None
        
    # Load each expert model
    expert_models = {}
    statuses = ['Salaried', 'Self-Employed', 'Student', 'Unemployed']
    for status in statuses:
        try:
            model_path = os.path.join(models_path, f"expert_model_{status}.joblib")
            expert_models[status] = joblib.load(model_path)
        except FileNotFoundError:
            st.error(f"Model file for '{status}' not found. Please ensure all models were saved.")
            return None, None
            
    return expert_models, train_cols

# --- Main Dashboard UI ---
st.set_page_config(page_title="Default Risk Predictor", layout="wide")
st.title("⚡ Interactive Default Risk Dashboard")
st.write("This app uses your pre-trained models to provide instant predictions based on customer data.")

# Load the models and column data
trained_models, train_columns = load_models()

# Sidebar for user inputs
st.sidebar.header("Enter Customer Information")

# Only show inputs if models were loaded successfully
if trained_models is not None:
    # --- Input Fields for all features in your dataset ---
    st.sidebar.subheader("Personal & Loan Details")
    employment_status = st.sidebar.selectbox('Employment Status', ('Salaried', 'Self-Employed', 'Student', 'Unemployed'))
    age = st.sidebar.slider('Age', 18, 70, 35)
    income = st.sidebar.number_input('Annual Income (INR)', min_value=0, value=500000, step=10000)
    loan_amount = st.sidebar.number_input('Loan Amount (INR)', min_value=1, value=100000, step=10000)
    tenure_months = st.sidebar.number_input('Loan Tenure (Months)', min_value=1, value=24, step=1)
    interest_rate = st.sidebar.slider('Interest Rate (%)', 1.0, 30.0, 12.5, 0.5)
    location = st.sidebar.selectbox('Location', ('Urban', 'Semi-Urban', 'Rural'))
    loan_type = st.sidebar.selectbox('Loan Type', ('Personal Loan', 'Home Loan', 'Education Loan', 'Business Loan', 'Vehicle Loan'))

    st.sidebar.subheader("Behavioral & History Data")
    missed_payments = st.sidebar.number_input('Number of Missed Payments', min_value=0, value=0)
    partial_payments = st.sidebar.number_input('Number of Partial Payments', min_value=0, value=1)
    delay_days = st.sidebar.number_input('Total Days Delayed', min_value=0, value=0)
    interaction_attempts = st.sidebar.number_input('Number of Interaction Attempts', min_value=0, value=5)
    sentiment_score = st.sidebar.slider('Average Sentiment Score', -1.0, 1.0, 0.2, 0.1)
    response_time_hours = st.sidebar.number_input('Average Response Time (Hours)', min_value=0, value=24)
    app_usage_frequency = st.sidebar.number_input('App Usage Frequency Score', min_value=0, value=10)
    website_visits = st.sidebar.number_input('Number of Website Visits', min_value=0, value=3)
    complaints = st.sidebar.number_input('Number of Complaints Registered', min_value=0, value=0)


    # Prediction Button
    if st.sidebar.button("Calculate Default Probability"):
        
        # --- Create Engineered Features from Inputs ---
        debt_to_income = loan_amount / income if income > 0 else 0
        repayment_score = missed_payments * 3 + partial_payments * 1 + delay_days * 0.2
        interaction_effectiveness = sentiment_score / interaction_attempts if interaction_attempts > 0 else 0
        
        # Create a dictionary with all base and engineered features
        input_data = {
            'Age': [age], 'Income': [income], 'LoanAmount': [loan_amount], 'TenureMonths': [tenure_months],
            'InterestRate': [interest_rate], 'MissedPayments': [missed_payments], 'PartialPayments': [partial_payments],
            'DelaysDays': [delay_days], 'InteractionAttempts': [interaction_attempts], 'SentimentScore': [sentiment_score],
            'ResponseTimeHours': [response_time_hours], 'AppUsageFrequency': [app_usage_frequency],
            'WebsiteVisits': [website_visits], 'Complaints': [complaints],
            'Debt_to_Income_Ratio': [debt_to_income], 'Repayment_Score': [repayment_score],
            'Interaction Effectiveness': [interaction_effectiveness],
            f'Location_{location}': [1], f'LoanType_{loan_type}': [1], f'EmploymentStatus_{employment_status}': [1]
        }
        input_df = pd.DataFrame(input_data)
        
        # Align columns with the training data to ensure order and presence match
        input_df_aligned = input_df.reindex(columns=train_columns, fill_value=0)
        
        # Select the correct expert model based on employment status
        expert_pipeline = trained_models[employment_status]

        # Prepare features for prediction (drop the one-hot encoded employment columns)
        employment_dummies = ['EmploymentStatus_Salaried', 'EmploymentStatus_Self-Employed', 'EmploymentStatus_Unemployed', 'EmploymentStatus_Student']
        features_to_predict = input_df_aligned.drop(employment_dummies, axis=1)

        # Make the prediction
        probability = expert_pipeline.predict_proba(features_to_predict)[:, 1][0]
        
        # Display the result
        st.subheader("Prediction Result")
        prob_percentage = probability * 100
        
        if prob_percentage >= 50:
            st.error(f"High Risk: {prob_percentage:.2f}% Probability of Default")
        else:
            st.success(f"Low Risk: {prob_percentage:.2f}% Probability of Default")
        
        st.progress(probability)
        st.write(f"The model for **{employment_status}** customers was used for this prediction.")
else:
    st.warning("Could not load models. Please ensure the `models` directory exists and run the saving script.")

