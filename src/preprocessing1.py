import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

X_train = pd.read_csv('data/processed/X_train_preprocessed.csv')
X_test = pd.read_csv('data/processed/X_test_preprocessed.csv')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PowerTransformer

import joblib
import json
import os

np.random.seed(42)
columns_to_scale = ['remark_char_length','remark_word_count','remark_sentence_count','sentiment_polarity','sentiment_subjectivity','time_to_issueHrs','IssueResponseTimeHrs','category_fe' ,'category_encoded' ,'sub_category_fe' ,'subcat_te' ,'item_price_log']

# Decide which need transformation: skew > |1| is highly skewed
to_transform = [col for col in columns_to_scale if abs(X_train[col].dropna().skew()) > 1.0]
print("Highly skewed features:", to_transform)

# Apply Yeo-Johnson to the skewed ones (handles zeros & negatives)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_trans = X_train.copy()
X_train_trans[to_transform] = pt.fit_transform(X_train[to_transform])

X_test_trans = X_test.copy()
X_test_trans[to_transform] = pt.transform(X_test[to_transform])

# save the list of transformed columns as json
with open("artifacts/preprocess/features_to_transform.json", "w") as f:
    json.dump(to_transform, f)
# save the powertrasformer object pt using joblib
joblib.dump(pt, "artifacts/preprocess/power_transformer.joblib")

to_scale = [col for col in columns_to_scale if col not in to_transform]

scaler = StandardScaler()
# scaling only the to_scale columns
X_train_trans_scaled = X_train_trans.copy()
X_test_trans_scaled = X_test_trans.copy()
X_train_trans_scaled[to_scale] = scaler.fit_transform(X_train_trans_scaled[to_scale])
X_test_trans_scaled[to_scale] = scaler.transform(X_test_trans_scaled[to_scale])

# save the scaler
joblib.dump(scaler, "artifacts/preprocess/scaler.joblib")
# save the final data
# save transformed datasets
X_train_trans_scaled.to_csv("data/processed/X_train_transformed.csv", index=False)
X_test_trans_scaled.to_csv("data/processed/X_test_transformed.csv", index=False)


# Checking drift in the data using evidently

# Auto-detecting version and importing correctly
try:
    # Try new API first
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    print("✅ Using Evidently v0.7+ (New API)")
    
except ImportError:
    try:
        # Fall back to legacy API
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
        print("✅ Using Evidently v0.6 (Legacy API)")
        
    except ImportError:
        print("❌ Evidently not installed. Run: pip install evidently")
# Create and run drift report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=X_train_trans_scaled, current_data=X_test_trans_scaled)

# Save interactive HTML report
drift_report.save_html("drift_report.html")



# creating validation split
X_val, X_test, y_val, y_test = train_test_split(X_test_trans_scaled, y_test, test_size=0.5, random_state=42)


os.makedirs("data/processed/train_val_test", exist_ok=True)
X_train_trans_scaled.to_csv("data/processed/train_val_test/X_train.csv", index=False)
X_val.to_csv("data/processed/train_val_test/X_val.csv", index=False)
X_test.to_csv("data/processed/train_val_test/X_test.csv", index=False)
# saving y_train , y_val , y_test
np.save("data/processed/train_val_test/y_train.npy", y_train)
np.save("data/processed/train_val_test/y_val.npy", y_val)
np.save("data/processed/train_val_test/y_test.npy", y_test)


# for balanced class training , implementing smote to create X_train_smote and X_test_smote 

# import smote
from imblearn.over_sampling import SMOTE

# Create SMOTE instance
smote = SMOTE(random_state=42)

# Fit and resample the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# converting to numpy array and saving the data
X_train_smote = X_train_smote.to_numpy()
y_train_smote = y_train_smote.to_numpy()

np.save("data/processed/train_val_test/X_train_smote.npy", X_train_smote)
np.save("data/processed/train_val_test/y_train_smote.npy", y_train_smote)


from imblearn.combine import SMOTETomek

# Create SMOTE-Tomek instance
smote_tomek = SMOTETomek(random_state=42)

# Fit and resample the training data
X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train, y_train)

X_train_smote_tomek = X_train_smote_tomek.to_numpy()
y_train_smote_tomek = y_train_smote_tomek.to_numpy()

np.save("data/processed/train_val_test/X_train_smote.npy", X_train_smote_tomek)
np.save("data/processed/train_val_test/y_train_smote.npy", y_train_smote_tomek)


