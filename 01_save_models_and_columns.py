import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

print("--- Starting Model Training and Saving Process ---")

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv("Analytics_loan_collection_dataset.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: `Analytics_loan_collection_dataset.csv` not found.")
    print("Please make sure the dataset file is in the same folder as this script.")
    exit()

# --- 2. Perform Your Feature Engineering (Corrected Version) ---
# This block now uses the exact column names you provided.
try:
    # NOTE: Using 'Income' and 'TenureMonths' as per your column list.
    df['Debt_to_Income_Ratio'] = df['LoanAmount'] / df['Income']
    df['Repayment_Score'] = df['MissedPayments']*3 + 1*df['PartialPayments'] + 0.2*df['DelaysDays']
    df['Interaction Effectiveness'] = np.where(df['InteractionAttempts'] == 0, 0, df['SentimentScore'] / df['InteractionAttempts'])
   
    
    print("Feature engineering complete.")

except KeyError as e:
    print(f"\n--- Feature Engineering Error ---")
    print(f"A column required for feature engineering is missing: {e}")
    print("Please ensure your CSV file contains all required columns for the calculations.")
    exit()

# --- Post-Engineering Processing ---
df.replace([np.inf, -np.inf], 0, inplace=True)
df.drop('CustomerID', axis=1, inplace=True)
df_dum = pd.get_dummies(df, columns=['Location', 'LoanType', 'EmploymentStatus'], drop_first=False)


# --- 3. Split Data ---
X = df_dum.drop('Target', axis=1)
y = df_dum['Target']
# Using the full dataset for training as per your original logic
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Data prepared with {X_train.shape[1]} features.")

# --- 4. Define and Train the "Expert" Models ---
expert_models_config = {
    'Student': RandomForestClassifier(random_state=42),
    'Salaried': MLPClassifier(random_state=42, max_iter=500),
    'Self-Employed': MLPClassifier(random_state=42, max_iter=500),
    'Unemployed': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

trained_experts = {}
employment_dummy_columns = ['EmploymentStatus_Salaried', 'EmploymentStatus_Self-Employed', 'EmploymentStatus_Unemployed', 'EmploymentStatus_Student']

for status, model in expert_models_config.items():
    print(f"--- Training expert for: {status} ---")
    dummy_col_name = f"EmploymentStatus_{status}"
    
    train_segment_mask = X_train[dummy_col_name] == 1
    X_train_segment = X_train[train_segment_mask]
    y_train_segment = y_train[train_segment_mask]

    # Skip if no data for this segment
    if len(X_train_segment) == 0:
        print(f"Skipping {status}, no training data available.")
        continue
        
    features_to_use = X_train_segment.drop(employment_dummy_columns, axis=1, errors='ignore')
    numerical_features = features_to_use.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features)], remainder='passthrough')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    pipeline.fit(features_to_use, y_train_segment)
    trained_experts[status] = pipeline
    print(f"Expert for '{status}' trained successfully.")

# --- 5. Save the Trained Models and Column Layout ---
# Create a directory to store the models
if not os.path.exists('models'):
    os.makedirs('models')

# Save each trained pipeline
for status, pipeline in trained_experts.items():
    model_filename = f"models/expert_model_{status}.joblib"
    joblib.dump(pipeline, model_filename)
    print(f"Saved model for '{status}' to {model_filename}")

# Save the list of columns the model was trained on
column_list = X_train.columns.tolist()
joblib.dump(column_list, 'models/training_columns.joblib')
print("Saved training column layout to models/training_columns.joblib")

print("\n✅ Process complete. Models are saved and ready for the dashboard.")

