Default Risk Prediction Dashboard
This project contains a Streamlit dashboard that predicts the probability of a customer defaulting on a loan. It uses a "Mixture of Experts" model, where a specific machine learning model is trained for each customer employment category.

This repository is structured for easy deployment on Streamlit Community Cloud.

🚀 How to Deploy Your Dashboard
Follow these steps to get your own live version of this dashboard running from your GitHub account.

Step 1: Create a New GitHub Repository
Go to your GitHub profile.

Click on the + icon in the top right and select "New repository".

Name your repository (e.g., loan-risk-dashboard).

Make sure it is set to Public.

Click "Create repository".

Step 2: Upload All Project Files
In your new repository, click the "Add file" button and then "Upload files".

Upload all of the following files into your repository:

README.md (this file)

01_save_models_and_columns.py

dashboard_optimized.py

requirements.txt

.gitignore

Your dataset file: Analytics_loan_collection_dataset.csv

Step 3: Run the Model Training Script (One-Time Local Setup)
The dashboard loads pre-trained models. You must create these model files locally first and then upload them.

Download your repository to your computer.

Open your terminal or command prompt.

Navigate into the project folder using the cd command (e.g., cd path/to/your/loan-risk-dashboard).

Install the required libraries by running:

pip install -r requirements.txt

Run the saving script to train and save the models:

python 01_save_models_and_columns.py

A new folder named models will be created. Upload this entire models folder to your GitHub repository.

Step 4: Deploy on Streamlit Community Cloud
Go to share.streamlit.io and sign in with your GitHub account.

Click the "New app" button.

Under "Repository," select the GitHub repository you just created.

The "Main file path" should automatically be set to dashboard_optimized.py. If not, select it.

Click "Deploy!".
