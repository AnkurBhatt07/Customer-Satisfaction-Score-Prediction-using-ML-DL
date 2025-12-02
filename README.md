📊 Customer CSAT Prediction System
End-to-End Machine Learning Project with Flask Deployment

This project predicts Customer Satisfaction (CSAT) for an eCommerce support system using an end-to-end ML pipeline, including:

Data cleaning

Extensive feature engineering

Custom preprocessing

Handling missing values

Encoding categorical columns

Scaling numerical variables

Outlier processing

Feature selection (Correlation + VIF-based reduction)

Model training

Saving all preprocessing artifacts

Building a real-time Flask web app for prediction

🚀 Final Deployed Model

The Flask app uses the final selected model:

lr5.joblib

Note : Multiple machine learning models were trained and evaluated during development (logistic regression variants, decision trees, random forests, XGBoost). The best performing model based on validation metrics was selected as lr5.Similarly Deep Learning models were experimented with but logistic regression provided the best balance of performance and interpretability for this tabular dataset.


This is a binary logistic regression model, predicting:

1 → Satisfied Customer

0 → Unsatisfied Customer

Note: Initially the dataset had 5 classes (1-5). These were binarized into 2 classes for simplicity.Since the main business need was to identify satisfied vs unsatisfied customers, this binarization helped improve model focus and performance.

🗂️ Project Structure
project/
│── app.py                          # Flask backend (prediction pipeline)
│── data_analysis1/artifacts/models/lr5.joblib                      # Final deployed ML model
│── templates/
│      ├── base.html
│      ├── index.html               # Input form with dynamic dropdowns
│      └── result.html              # Prediction output page
│
│── data_analysis1/artifacts/preprocess/  # Preprocessing artifacts 
│          ├── unique_values.json
│          ├── mapped_cities.json
│          ├── CityBinsScaler.joblib
│          ├── connectedTimebins_scaler.joblib
│          ├── product_ohe.joblib
│          ├── category_ohe.joblib
│          ├── subcategory_ohe.joblib
│          ├── channel_ohe.joblib
│          ├── supervisor_ohe.joblib
│          ├── manager_ohe.joblib
│          ├── shift_ohe.joblib
│          ├── remark_word_count_scaler.joblib
│          ├── price_transformer.joblib
│          ├── price_limits.joblib
│          ├── tenure_oe.joblib
│          ├── tenure_scaler.joblib
│          ├── agent_rating.joblib
│          ├── agent_bins_scaler.joblib
│          ├── response_time_fallback.joblib
│          ├── timeToIssue_limits.joblib
│          ├── data_matrix_drop_cols.joblib
│
└── README.md

📘 Project Overview

The goal is to predict Customer Satisfaction (CSAT Score) after an interaction with customer support.

The raw dataset contained:

Customer demographics

Product information

Interaction metadata

Support process timestamps

Agent/team performance

Customer remarks

Due to heavy missingness (up to 80%), the project required intensive preprocessing.

🧹 Data Cleaning & Preprocessing (Summary)

Your preprocessing pipeline (from data_analysis1.ipynb) included:

✔ Missing Value Indicators (MI)

For multiple columns (City, Connected Handling Time, Product Category, Remarks, etc.)

✔ Binning + Normalization

CityBins_normalized

ConnectedTimeBins_normalized

✔ Random sampling imputation for:

Item Price (preserving distribution)

Time-to-issue & Issue response time (fallback normal distribution)

✔ PowerTransformer for:

Item_Price_normalized

✔ One-Hot Encoding (drop='first') for:

Product Category

Channel

Category

Sub-category

Supervisor

Manager

Shift

✔ Ordinal Encoding + Scaling

Tenure Bucket → Tenure_encoded

✔ Text Features from Remarks

Sentiment (TextBlob)

Word count (scaled)

✔ Agent Performance Encoding

Median CSAT per agent → scaled

✔ Date/Time Feature Engineering

time_to_issue_hours

IssueResponseTimeHours (scaled)

✔ Feature Selection

Performed using:

Correlation thresholding

Variance Inflation Factor (VIF)
Artifacts saved as:

data_matrix_drop_cols.joblib

✔ Final Processed Matrix

The model was trained on:

data_matrix1.npy


Which contains all engineered + encoded + scaled features.

🧠 Model Training

Algorithm: Logistic Regression

Balanced dataset by class weighting

Several LR variants were trained → best performing = lr5

Saved as:

lr5.joblib


Used for real-time prediction in Flask.

🌐 Flask Web Application

The web app provides:

✔ Dynamic Dropdowns from unique_values.json

No hardcoding of lookup values.

✔ Automated Preprocessing

Exactly replicates notebook logic:

Encoding

Scaling

Text sentiment extraction

Time difference calculation

Binning + normalization

Fallback logic for invalid datetime entries

✔ Final Prediction

On the result page, the model returns:

Prediction:

"Satisfied Customer"

"Unsatisfied Customer"

Confidence:

e.g., 63.55%


▶️ How Prediction Works (End-to-End)

User selects values in the form

Flask receives the raw inputs

Flask applies all preprocessing transformations using saved artifacts

Constructs the exact feature row expected by lr5.joblib

Passes the row to the model

Displays prediction and satisfaction Probability on result page


🙌 Credits

This project was built by Ankur Bhatt using:

Python

Pandas

Numpy

Scikit-learn

Flask

Joblib

TextBlob

Matplotlib / Seaborn