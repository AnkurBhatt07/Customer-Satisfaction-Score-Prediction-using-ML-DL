from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import json
from textblob import TextBlob
import warnings 
warnings.filterwarnings('ignore')

# ------------------------
# Load Preprocessing Artifacts
# ------------------------
with open("artifacts/preprocess/city_counts.json", "r") as f:
    city_counts = json.load(f)

with open("artifacts/preprocess/mapped_cities.json", "r") as f:
    mapped_cities = json.load(f)

# Load unique values for dropdowns (you'll need to create these)
with open("artifacts/preprocess/unique_values.json", "r") as f:
    unique_values = json.load(f)

# Load all preprocessing artifacts (same as your original code)
CityBinsScaler = joblib.load("artifacts/preprocess/CityBinsScaler.joblib")
connectedTimeBinsScaler = joblib.load("artifacts/preprocess/connectedTimebins_scaler.joblib")
product_ohe = joblib.load("artifacts/preprocess/product_ohe.joblib")
pt_price = joblib.load("artifacts/preprocess/price_transformer.joblib")
remark_word_count_scaler = joblib.load("artifacts/preprocess/remark_word_count_scaler.joblib")
channel_ohe = joblib.load("artifacts/preprocess/channel_ohe.joblib")
category_ohe = joblib.load("artifacts/preprocess/category_ohe.joblib")
sub_cat_ohe = joblib.load("artifacts/preprocess/subcategory_ohe.joblib")
agent_rating = joblib.load("artifacts/preprocess/agent_rating.joblib")
agent_bins_scaler = joblib.load("artifacts/preprocess/agent_bins_scaler.joblib")
supervisor_ohe = joblib.load("artifacts/preprocess/supervisor_ohe.joblib")
supervisor_rating_scaler = joblib.load("artifacts/preprocess/supervisor_rating_scaler.joblib")
supervisor_rating = joblib.load("artifacts/preprocess/supervisor_rating.joblib")
manager_ohe = joblib.load("artifacts/preprocess/manager_ohe.joblib")
tenure_oe = joblib.load("artifacts/preprocess/tenure_oe.joblib")
tenure_scaler = joblib.load("artifacts/preprocess/tenure_scaler.joblib")
shift_ohe = joblib.load("artifacts/preprocess/shift_ohe.joblib")

# Load the trained model
model = joblib.load("artifacts/models/lr4_binary_classification_model.joblib")

# ------------------------
# Flask app
# ------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key

# ------------------------
# Helper Functions (same as your original)
# ------------------------
def get_sentiment(text):
    if pd.isna(text):
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def remark_word_count(text):
    if pd.isna(text):
        return 0
    return len(text.split())

def city_bin(city):
    counts = city_counts.get(city, city_counts.get("NaN", 0))
    if counts < 50: return 1
    elif counts < 100: return 2
    elif counts < 200: return 3
    elif counts < 400: return 4
    elif counts < 800: return 5
    else: return 6

def connectedTimebins(time):
    if pd.isna(time): return 6
    if time < 250: return 1
    elif time < 500: return 2
    elif time < 750: return 3
    elif time < 1000: return 4
    else: return 5

# ------------------------
# Routes
# ------------------------
@app.route('/')
def home():
    """
    This route renders the main page with the prediction form.
    It passes all the unique values from training data to create dropdown options.
    """
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict_form():
    """
    This route handles form submission from the web interface.
    It validates the input, processes the data, and returns the prediction.
    """
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate that all categorical inputs are from allowed values
        validation_errors = []
        
        if form_data.get('Customer_City') not in unique_values['cities']:
            validation_errors.append(f"Invalid city: {form_data.get('Customer_City')}")
        
        if form_data.get('Product_category') not in unique_values['products']:
            validation_errors.append(f"Invalid product category: {form_data.get('Product_category')}")
        
        if form_data.get('channel_name') not in unique_values['channels']:
            validation_errors.append(f"Invalid channel: {form_data.get('channel_name')}")
        
        if form_data.get('category') not in unique_values['categories']:
            validation_errors.append(f"Invalid category: {form_data.get('category')}")
        
        if form_data.get('Sub-category') not in unique_values['sub_categories']:
            validation_errors.append(f"Invalid sub-category: {form_data.get('Sub-category')}")
        
        if form_data.get('Agent_name') not in unique_values['agents']:
            validation_errors.append(f"Invalid agent: {form_data.get('Agent_name')}")
        
        if form_data.get('Supervisor') not in unique_values['supervisors']:
            validation_errors.append(f"Invalid supervisor: {form_data.get('Supervisor')}")
        
        if form_data.get('Manager') not in unique_values['managers']:
            validation_errors.append(f"Invalid manager: {form_data.get('Manager')}")
        
        if form_data.get('Tenure Bucket') not in unique_values['tenure_buckets']:
            validation_errors.append(f"Invalid tenure bucket: {form_data.get('Tenure Bucket')}")
        
        if form_data.get('Agent Shift') not in unique_values['shifts']:
            validation_errors.append(f"Invalid shift: {form_data.get('Agent Shift')}")
        
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
            return redirect(url_for('home'))
        
        # Convert form data to DataFrame
        df = pd.DataFrame([form_data])
        
        # Convert numeric fields
        df['connected_handling_time'] = pd.to_numeric(df['connected_handling_time'], errors='coerce')
        df['Item_price'] = pd.to_numeric(df['Item_price'], errors='coerce')
        df['time_to_issue_hours'] = pd.to_numeric(df['time_to_issue_hours'], errors='coerce')
        df['IssueResponseTimeHours'] = pd.to_numeric(df['IssueResponseTimeHours'], errors='coerce')
        
        # Apply the same preprocessing as your original predict route
        # [Include all your original preprocessing code here]
        # Feature Engineering
        df["City_MI"] = df["Customer_City"].isna().astype(int)
        df['CityBins'] = df['Customer_City'].map(mapped_cities)
        df['CityBins_normalized'] = CityBinsScaler.transform(df[["CityBins"]])

        df['connectedTime_MI'] = df['connected_handling_time'].isna().astype(int)
        df['ConnectedTimeBins'] = df['connected_handling_time'].apply(connectedTimebins)
        df['connectedTimeBins_normalized'] = connectedTimeBinsScaler.transform(df[["ConnectedTimeBins"]])

        df['Product_cat_MI'] = df['Product_category'].isna().astype(int)
        df['Product_category'] = df['Product_category'].fillna("Unknown")
        product_encoded = product_ohe.transform(df[['Product_category']])

        df['Price_MI'] = df['Item_price'].isna().astype(int)
        df['Item_price'] = df['Item_price'].fillna(np.mean(df['Item_price'].dropna()))
        df['Item_Price_normalized'] = pt_price.transform(df[['Item_price']])

        df['Remarks_MI'] = df['Customer_Remarks'].isna().astype(int)
        df['Remarks_sentiment'] = df['Customer_Remarks'].apply(get_sentiment)
        df['Remark_word_count'] = df['Customer_Remarks'].apply(remark_word_count)
        df['Remark_word_count'] = remark_word_count_scaler.transform(df[['Remark_word_count']])

        channel_encoded = channel_ohe.transform(df[['channel_name']])
        categories_encoded = category_ohe.transform(df[['category']])
        sub_categories_encoded = sub_cat_ohe.transform(df[['Sub-category']])

        df['agent_bins'] = df['Agent_name'].map(agent_rating)
        df['agent_bins'] = agent_bins_scaler.transform(df[['agent_bins']])

        df['supervisor_rating'] = df['Supervisor'].map(supervisor_rating)
        df['supervisor_rating'] = supervisor_rating_scaler.transform(df[['supervisor_rating']])

        manager_encoded = manager_ohe.transform(df[['Manager']])

        df['Tenure_encoded'] = tenure_oe.transform(df[['Tenure Bucket']])
        df['Tenure_encoded'] = tenure_scaler.transform(df[['Tenure_encoded']])

        shift_encoded = shift_ohe.transform(df[['Agent Shift']])

        # Create data matrix (same as your original code)
        data_matrix = df.loc[:,['City_MI', 'CityBins_normalized', 'connectedTime_MI',
           'connectedTimeBins_normalized', 'Product_cat_MI']].to_numpy()
        data_matrix = np.concatenate([data_matrix, product_encoded], axis=1)
        data_matrix = np.concatenate([data_matrix, df[['Item_Price_normalized']].values], axis=1)
        data_matrix = np.concatenate([data_matrix, df[["Remarks_MI"]].values], axis=1)
        data_matrix = np.concatenate([data_matrix, df[['Remarks_sentiment']].values], axis=1)
        data_matrix = np.concatenate([data_matrix, df[['Remark_word_count']].values], axis=1)
        data_matrix = np.concatenate([data_matrix, channel_encoded], axis=1)
        data_matrix = np.concatenate([data_matrix, categories_encoded], axis=1)
        data_matrix = np.concatenate([data_matrix, sub_categories_encoded], axis=1)
        data_matrix = np.concatenate([data_matrix, df[["time_to_issue_hours"]].values], axis=1)
        data_matrix = np.concatenate([data_matrix, df[['IssueResponseTimeHours']].values], axis=1)
        data_matrix = np.concatenate([data_matrix, df[['agent_bins']].values], axis=1)
        data_matrix = np.concatenate([data_matrix, df[["supervisor_rating"]].values], axis=1)
        data_matrix = np.concatenate([data_matrix, manager_encoded], axis=1)
        data_matrix = np.concatenate([data_matrix, df[["Tenure_encoded"]].values], axis=1)
        data_matrix = np.concatenate([data_matrix, shift_encoded], axis=1)
        
        # Make prediction
        pred = model.predict(data_matrix)
        pred_class = "CSAT >= 3" if pred == 1 else "CSAT < 3"
        
        # Render result page with prediction
        return render_template('result.html', 
                             prediction=pred_class, 
                             form_data=form_data)
        
    except Exception as e:
        flash(f"Error processing prediction: {str(e)}", 'error')
        return redirect(url_for('home'))



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
