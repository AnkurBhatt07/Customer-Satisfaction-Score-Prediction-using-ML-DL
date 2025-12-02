# 📊 Customer CSAT Prediction System
### End-to-End Machine Learning Project with Flask Deployment

🔗 **GitHub Repository:**  
https://github.com/AnkurBhatt07/Customer-Satisfaction-Score-Prediction-using-ML-DL.git  

---

## 🚀 Overview

This project predicts **Customer Satisfaction (CSAT)** for an eCommerce support system using a complete end-to-end Machine Learning pipeline and a real-time **Flask web application**.

The final model classifies customers as:

- **1 → Satisfied Customer**  
- **0 → Unsatisfied Customer**

Originally, the dataset contained CSAT levels 1-5, but these were binarized (satisfied vs unsatisfied) to align with the business requirement and improve model performance.

---

## 🧠 Key Features

✔ End-to-end ML pipeline  
✔ Real-time Flask prediction app  
✔ Production-ready preprocessing using saved artifacts  
✔ Extensive feature engineering  
✔ Interpretability-focused Logistic Regression model  
✔ Dynamic web form (dropdowns generated automatically from JSON)

---

## 🗂️ Project Structure

```
project/
├── app.py                                   # Flask backend (prediction pipeline)
│
├── data_analysis1/
│   ├── artifacts/
│   │   ├── models/
│   │   │   └── lr5.joblib                   # Final deployed ML model
│   │   │
│   │   ├── preprocess/                      # Preprocessing artifacts
│   │       ├── unique_values.json
│   │       ├── mapped_cities.json
│   │       ├── CityBinsScaler.joblib
│   │       ├── connectedTimebins_scaler.joblib
│   │       ├── product_ohe.joblib
│   │       ├── category_ohe.joblib
│   │       ├── subcategory_ohe.joblib
│   │       ├── channel_ohe.joblib
│   │       ├── supervisor_ohe.joblib
│   │       ├── manager_ohe.joblib
│   │       ├── shift_ohe.joblib
│   │       ├── remark_word_count_scaler.joblib
│   │       ├── price_transformer.joblib
│   │       ├── price_limits.joblib
│   │       ├── tenure_oe.joblib
│   │       ├── tenure_scaler.joblib
│   │       ├── agent_rating.joblib
│   │       ├── agent_bins_scaler.joblib
│   │       ├── response_time_fallback.joblib
│   │       ├── timeToIssue_limits.joblib
│   │       └── data_matrix_drop_cols.joblib
│   │
│   └── data_matrix1.npy                     # Final processed training matrix
│
├── templates/
│   ├── base.html
│   ├── index.html                           # Input form with dynamic dropdowns
│   └── result.html                          # Prediction output page
│
└── README.md
```

---

## 🧹 Data Cleaning & Preprocessing

### Highlights:
- Missing value indicators  
- Binning + normalization  
- Smart imputation  
- PowerTransformer  
- One-hot encoding  
- Ordinal encoding + scaling  
- Text sentiment features  
- Agent performance encoding  
- Date/time features  
- Correlation & VIF-based feature selection  

---

## 🤖 Model Training

Algorithms tested: Logistic Regression, Decision Tree, Random Forest, XGBoost, ANN.  
👉 **Best model:** `lr5.joblib` (Logistic Regression)

---

## 🌐 Flask Web App

✔ Dynamic dropdown generation  
✔ Fully automated preprocessing  
✔ Probability-based CSAT prediction  
✔ Clean UI

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, Sklearn, Flask, Joblib, TextBlob, Matplotlib, Seaborn  

---

## 🙌 Credits  
Built with ❤️ by **Ankur Bhatt**.
