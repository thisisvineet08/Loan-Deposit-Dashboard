#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load Data From CSV File ---
print("Loading data from loan_data.csv...")
try:
    df = pd.read_csv('Loan_Entry.csv')
except FileNotFoundError:
    print("Error: 'loan_data.csv' not found. Please make sure the CSV file is in the same folder as this script.")
    exit()

# --- 2. Prepare the Data for Modeling (New Customer Focus) ---
print("Preparing data for NEW CUSTOMER model...")
# Drop columns not needed or not suitable for this model
df_prepared = df.drop(columns=['CustomerID', 'Location','MissedPayments','DelaysDays','PartialPayments','InteractionAttempts','SentimentScore','ResponseTimeHours','AppUsageFrequency','WebsiteVisits','Complaints'])

# *** Select ONLY features available for a NEW customer ***
core_features = [
    'Age',
    'Income',
    'EmploymentStatus',
    'LoanAmount',
    'TenureMonths',
    'InterestRate',
    'Target' # Keep target for splitting
]
df_model = df_prepared[core_features].copy()


# In[2]:


from sklearn.ensemble import RandomForestClassifier


# In[3]:


from sklearn.model_selection import train_test_split, GridSearchCV


# In[4]:


# Feature Engineering on the selected data
df_model['DTI_Ratio'] = df_model['LoanAmount'] / df_model['Income']
df_model['Loan_Burden_Index'] = (df_model['LoanAmount'] * df_model['InterestRate']) / df_model['TenureMonths']

# *** THE FIX: Drop raw features to force model to use the more meaningful engineered ones ***
df_model = df_model.drop(columns=['LoanAmount', 'Income'])


X = df_model.drop(columns=['Target'])
y = df_model['Target']

# Switched to drop_first=False to ensure all categories are handled consistently
X_encoded = pd.get_dummies(X, columns=['EmploymentStatus'], drop_first=True)

# --- 3. Split Data and Find Best Model with Grid Search ---
print("Splitting data and finding the best model parameters with Grid Search...")
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# *** THE FIX: Reduced the number of options to make the search run faster ***
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the Grid Search with a Random Forest model
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1 # Shows progress
)

# Fit Grid Search to the training data
grid_search.fit(X_train, y_train)

# Get the best model found by Grid Search
best_model = grid_search.best_estimator_
print(f"\nBest parameters found: {grid_search.best_params_}")

# --- 4. Check the Accuracy of the Best Model ---
print("\n--- Model Evaluation (using best model) ---")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("------------------------\n")

# --- 5. Train Final Model and Save ---
print("Training the final model on all data using the best parameters...")
final_model = grid_search.best_estimator_
final_model.fit(X_encoded, y)

# Save the model and columns to files
joblib.dump(final_model, 'risk_model.pkl')
model_columns = list(X_encoded.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("Final 'New Customer' model (Optimized Random Forest) and columns have been saved successfully!")
print("You can now run the 'dashboard.py' file.")



# In[ ]:




