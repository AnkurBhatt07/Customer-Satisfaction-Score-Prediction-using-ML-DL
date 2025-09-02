# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import joblib, json, time
import os
import warnings
warnings.filterwarnings("ignore")
from textblob import TextBlob
# %%
import sys 
import os
# sys.path.append(os.path.abspath(".."))


# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from src.utils import CustomerRemarksPreprocessor

# %%
RAW_CSV = r"data\raw\eCommerce_Customer_support_data.csv"   # adjust to your layout
ARTIFACTS_DIR = r"artifacts"
PREPROCESS_DIR = rf"{ARTIFACTS_DIR}\preprocess"
MODELS_DIR = rf"{ARTIFACTS_DIR}\models"


# %%
# import os
for d in (ARTIFACTS_DIR, PREPROCESS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


# %%
data = pd.read_csv(RAW_CSV)

# %%
# data.info()

# %%
data.drop(columns = ['Unique id' , 'Order_id' ],axis= 1 , inplace = True)


# %%
data['Customer Remarks'].fillna("<Unknown>" , inplace = True)

data['remark_char_length'] = data['Customer Remarks'].str.len()
data['remark_word_count'] = data['Customer Remarks'].str.split().str.len()
data['remark_sentence_count'] = data['Customer Remarks'].str.split('.').str.len()

# import sys
# sys.path.append(os.path.abspath(".."))

preprocessor = CustomerRemarksPreprocessor()
data['cleaned_remarks'] = data['Customer Remarks'].apply(preprocessor.preprocess)


# creating has_missing_remarks column
data['has_missing_remarks'] = data['cleaned_remarks'].apply(lambda x: 1 if x == '<Unknown>' else 0)

# Replacing the remark_char_length , remark_word_count , remark_sentence_count for records having cleaned_remarks value as <Missing_Remark> with -1
data.loc[data['cleaned_remarks'] == '<Missing_Remark>', 'remark_char_length'] = -1
data.loc[data['cleaned_remarks'] == '<Missing_Remark>', 'remark_word_count'] = -1
data.loc[data['cleaned_remarks'] == '<Missing_Remark>', 'remark_sentence_count'] = -1

# %%
# from textblob import TextBlob
# import pandas as pd


def get_sentiment_scores(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

remarks = data['Customer Remarks'].replace({'<Unknown>':""})
scores = remarks.apply(get_sentiment_scores)


# %%
# Convert tuple series to DataFrame
scores_df = pd.DataFrame(scores.tolist(), columns=["sentiment_polarity", "sentiment_subjectivity"])

# Join back to data
data = pd.concat([data, scores_df], axis=1)

# %%
data['order_date_time'] = pd.to_datetime(data['order_date_time'], format='%d/%m/%Y %H:%M',errors='coerce')
data['Issue_reported at'] = pd.to_datetime(data['Issue_reported at'], format='%d/%m/%Y %H:%M',errors='coerce')
data['issue_responded'] = pd.to_datetime(data['issue_responded'], format='%d/%m/%Y %H:%M',errors='coerce')

available_datetime = data[data['order_date_time'].notna()].copy()
available_datetime['order_date_time'] = pd.to_datetime(available_datetime['order_date_time'], 
                                                          format='%d/%m/%Y %H:%M')
available_datetime['Issue_reported at'] = pd.to_datetime(available_datetime['Issue_reported at'], 
                                                          format='%d/%m/%Y %H:%M')
available_datetime["time_to_issue"] = available_datetime['Issue_reported at'] - available_datetime['order_date_time']
available_datetime["time_to_issue_hours"] = available_datetime["time_to_issue"].dt.total_seconds() / 3600
avg_hours_diff = available_datetime["time_to_issue_hours"].median()


def impute_order_datetime(row):
    if pd.notna(row['order_date_time']):
        return row['order_date_time']
    elif pd.notna(row['Issue_reported at']):
        
        # Subtract average hours difference between order and issue reported
        hours_before = avg_hours_diff  # Use average hours difference calculated earlier
        return row['Issue_reported at'] - timedelta(hours=hours_before)

data['order_date_time_imputed'] = data.apply(impute_order_datetime, axis=1)

# Create missing indicator for non missing orderdate
data['has_order_date'] = data['order_date_time'].notna().astype(int)
# Feature engineering 
data['time_to_issue'] = data['issue_responded'] - data['Issue_reported at']
data['time_to_issueHrs'] = data['time_to_issue'].dt.total_seconds() / 3600  # Convert to hours

# saving the artifacts
os.makedirs(r"..\artifacts\preprocess", exist_ok=True)
with open(r"..\artifacts\preprocess\order_time_impute_params_v1.json", "w") as f:
    json.dump({"avg_hours_diff": avg_hours_diff}, f, indent=2)                                                                                                                


# %%
# feature engineering : difference between issue reported date and issue responded date
data['issue_responded'] = data['issue_responded'] + pd.Timedelta(hours=23, minutes=59, seconds=59)
data['IssueResponseTime'] = data['issue_responded'] - data['Issue_reported at']
data['IssueResponseTimeHrs'] = data['IssueResponseTime'].dt.total_seconds() / 3600  # Convert to hours


# %%
data['Customer_City'].fillna('<Unknown>', inplace=True)


# %%
# Replace the missing null values with "Unknown" Product category


data['Product_category'].fillna("<Unknown>" , inplace = True )

# %%
data['has_item_price'] = data['Item_price'].notna().astype(int)

# Zero price indicator  
data['is_zero_price'] = (data['Item_price'] == 0).astype(int)

#  Price categories
def categorize_price(price):
    if pd.isna(price):
        return '<Unknown>'
    elif price == 0:
        return 'Free'
    elif price <= 500:
        return 'Low'
    elif price <= 2000:
        return 'Medium'
    elif price <= 10000:
        return 'High'
    else:
        return 'Premium'

data['price_category'] = data['Item_price'].apply(categorize_price)



data['price_category_encoded'] = data['price_category'].map({
    '<Unknown>': -1,
    "Free": -1,
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Premium': 3
})


# %%


data["missing_connected_handling_time_flag"] = data['connected_handling_time'].isna().astype(int)

# Time-based categories
def categorize_handling_time(time):
    if pd.isna(time):
        return '<Unknown>'
    elif time == 0:
        return 'No_Interaction'
    elif time <= 300:
        return 'Quick'      # ≤5 minutes
    elif time <= 600:
        return 'Normal'     # 5-10 minutes
    elif time <= 1200:
        return 'Long'       # 10-20 minutes
    else:
        return 'Very_Long'  # >20 minutes

data['handling_time_category'] = data['connected_handling_time'].apply(categorize_handling_time)

# Ordinal Encoder function for handling time categories
def encoder(category):
    return category.map({
        '<Unknown>': 0,
        'No_Interaction': 0,
        'Quick': 1,
        'Normal': 2,
        'Long': 3,
        'Very_Long': 4
    })

data['handling_time_category_encoded'] = encoder(data['handling_time_category'])

# %%
data.drop(columns=['Agent_name', 'Supervisor', 'Manager'], axis = 1,inplace=True)


# %%
# Ordinal encoding tenure bucket column
# Convert to experience level (0-4)
tenure_mapping = {
    'On Job Training': 0,  # Least experienced
    '0-30': 1, 
    '31-60': 2,
    '61-90': 3,
    '>90': 4  # Most experienced
}
data['tenure_encoded'] = data['Tenure Bucket'].map(tenure_mapping)


# %%
# Train test split

from sklearn.model_selection import train_test_split

X = data.drop(columns=['CSAT Score'],axis = 1)
y = data['CSAT Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# saving these X and y df and series respectively in data/raw.currently in working in notebooks/preprocessing.ipynb.both notebooks and data folders are at the same level

# X.to_csv(r"data\raw\X.csv", index=False)
# y.to_csv(r"data\raw\y.csv", index=False)

# %%
# channel_name one hot encoding with drop = first



all_channels = sorted(data['channel_name'].dropna().unique())  # or a curated list
print(all_channels)

# CHANNEL_NAME OneHotEncoder
ohe = OneHotEncoder(categories=[all_channels], handle_unknown="ignore")
ohe.fit(X_train[['channel_name']])
print("Learned categories:", ohe.categories_)
feature_names = ohe.get_feature_names_out(['channel_name'])
print("Feature names:", feature_names)


X_train_ohe1 = ohe.transform(X_train[['channel_name']])
X_test_ohe1 = ohe.transform(X_test[['channel_name']])

# saving the ohe object as artifact after training it on train data and then transforming the train and the test data

joblib.dump(ohe, f"{PREPROCESS_DIR}\\channel_name_ohe_v1.joblib") 

# %%
# frequency encoding of category column

freq_map = X_train['category'].value_counts().to_dict()
X_train['category_fe'] = X_train['category'].map(freq_map)
X_test['category_fe'] = X_test['category'].map(freq_map)

# saving the mapping object as artifact after training it on train data and then transforming the train and the test data
joblib.dump(freq_map, f'{PREPROCESS_DIR}/category_freq_map_v1.joblib')


# combining the four lowest frequency categories into 'Others' for train and test data.
# .replace(['Offers & Cashback', 'Onboarding related', 'App/website'], 'Others')
X_train['category'].replace(['Offers & Cashback', 'Onboarding related', 'App/website'], 'Others', inplace=True)
X_test['category'].replace(['Offers & Cashback', 'Onboarding related', 'App/website'], 'Others', inplace=True)


# %%
# regularized target encoding of category column

def fit_regularized_target_encoding(X_train , ytrain , col = "category" , min_samples=1000, pull_strength=0.3):
    train_data = X_train.copy()
    train_data['target'] = ytrain
    cat_means = train_data.groupby([col])['target'].mean()
    global_mean = ytrain.mean()
    print("Global CSAT mean :" ,global_mean)

    final_encoding = {}
    for cat , mean in cat_means.items():
        count = len(train_data[train_data[col] == cat])
        if count < min_samples:
            regularized_mean = mean * (1-pull_strength) + global_mean * pull_strength
            print(f"{cat}: {mean:.2f} → {regularized_mean:.2f} (regularized)")
        else:
            regularized_mean = mean
            print(f"{cat}: {mean:.2f} → {regularized_mean:.2f} (trusted)")
        
        final_encoding[cat] = regularized_mean
    
    return final_encoding , global_mean



def apply_regularized_encoding(X, encoding_map, global_mean, category_col='category', encoded_col='category_encoded'):
    X = X.copy()
    X[encoded_col] = X[category_col].map(encoding_map)
    X[encoded_col] = X[encoded_col].fillna(global_mean)
    return X

encoding_map, global_mean = fit_regularized_target_encoding(X_train, y_train, col='category')

X_train = apply_regularized_encoding(X_train, encoding_map, global_mean, category_col='category')

X_test = apply_regularized_encoding(X_test, encoding_map, global_mean, category_col='category')



# saving the artifacts 

joblib.dump(encoding_map, f'{PREPROCESS_DIR}/category_te_map_v1.joblib')

# save global_mean as json
with open(fr"{PREPROCESS_DIR}\global_csat_mean.json", "w") as f:
    json.dump({"global_mean": global_mean}, f, indent=2)

# %%
# frequency encoding of sub-category
freq_map = X_train['Sub-category'].value_counts().to_dict()
X_train['sub_category_fe'] = X_train['Sub-category'].map(freq_map)
X_test['sub_category_fe'] = X_test['Sub-category'].map(freq_map)

# saving the freqmap
joblib.dump(freq_map , fr"{PREPROCESS_DIR}/sub_category_fe_map.joblib")



# Low count/frequency sub_category replacement with Others 
# low_subcat
vc = X_train['Sub-category'].value_counts()
low_subcat = vc[vc < 300].index
X_train['Sub-category'] = X_train['Sub-category'].replace(low_subcat, 'Others') 
X_test['Sub-category'] = X_test['Sub-category'].replace(low_subcat, 'Others') 

# saving the low_subcat
joblib.dump(low_subcat , fr"{PREPROCESS_DIR}/low_freq_sub_categories_list.joblib")

# %%
# Target Encoding for sub_category : KFOLD Targetencoding

def sub_cat_te_trainer(X_train:pd.DataFrame , y_train:pd.Series , n_splits:int , random_state:int , shuffle:bool , col_name : str = "Sub-category" , target_name : str = "CSAT Score" , encoded_col:str = "subcat_te") :
    Xtr = X_train.copy()
    ytr = y_train.reset_index(drop = True)
    Xtr = Xtr.reset_index(drop = True)

    Xtr[encoded_col] = np.nan

    kf = KFold(n_splits = n_splits ,shuffle = shuffle , random_state = random_state )

    for train_idx , val_idx in kf.split(Xtr):
        Xtr_fold = Xtr.iloc[train_idx]
        ytr_fold = ytr.iloc[train_idx]

        Xval_fold = Xtr.iloc[val_idx]

        df = pd.DataFrame({col_name : Xtr_fold[col_name] , target_name : ytr_fold})

        means = df.groupby(col_name)[target_name].mean().to_dict()

        Xtr.loc[val_idx , encoded_col] = Xval_fold[col_name].map(means)
    
    global_mean = ytr.mean()

    Xtr[encoded_col].fillna(global_mean , inplace = True)

    # Now the final mapping

    df = pd.DataFrame({col_name : Xtr[col_name] , target_name : ytr})

    final_mapping = df.groupby(col_name)[target_name].mean().to_dict()

    return Xtr , global_mean , final_mapping

# Trained te encoder map using X_train

X_train , global_csat_mean , final_mapping = sub_cat_te_trainer(X_train , y_train ,n_splits = 5 , random_state = 42 , shuffle = True )

# Implementing the  subcat_te on X_test

X_test['subcat_te'] = X_test['Sub-category'].map(final_mapping)

# saving the final mapping

joblib.dump(final_mapping,fr"{PREPROCESS_DIR}/sub_cat_te.joblib")


# %%


# Join X_train with y_train temporarily
train_with_target = X_train.copy()
train_with_target["CSAT Score"] = y_train

# Compute mean CSAT per city
city_mean_csat_scores = train_with_target.groupby("Customer_City")["CSAT Score"].mean()

# Convert means → category (1–5)
def city_map(score):
    if score <= 1:
        return 1
    elif score <= 2:
        return 2
    elif score <= 3:
        return 3
    elif score <= 4:
        return 4
    else:
        return 5

X_train['city_csat_mean'] = X_train['Customer_City'].map(city_mean_csat_scores)
X_train['city_category'] = X_train['city_csat_mean'].map(city_map)

X_test['city_csat_mean'] = X_test['Customer_City'].map(city_mean_csat_scores)
X_test['city_category'] = X_test['city_csat_mean'].map(city_map)

# if X_test gets some new Customer_City
X_test.loc[X_test['city_category'].isna() , 'city_category'] = city_map(city_mean_csat_scores['<Unknown>'])
# # Apply mapping to train and test
# city_mean_csat_scores['city_category'] = city_mean_csat_scores['CSAT Score'].apply(city_map)

# X_train['city_category'] = X_train["Customer_City"].map(city_mean_csat_scores.set_index('Customer_City')['city_category'])
# X_test['city_category'] = X_test["Customer_City"].map(city_mean_csat_scores.set_index('Customer_City')['city_category'])

# # Save the mapping for inference
joblib.dump(city_mean_csat_scores, f"{PREPROCESS_DIR}/city_mean_csat_scores_map.joblib")
print("City → category mapping saved.")


# %%
medians_by_cat = X_train.groupby('Product_category')['Item_price'].median().to_dict()
global_median = float(X_train['Item_price'].median())

# # ensure a fallback exists
medians_by_cat_fallback = medians_by_cat.copy()
medians_by_cat_fallback['__FALLBACK__'] = global_median

# import json, os
# os.makedirs(r"..\artifacts\preprocess", exist_ok=True)
with open(r"..\artifacts\preprocess\price_impute_params_v1.json","w") as f:
    json.dump({"global_price_median": global_median,
               "price_medians_by_category": medians_by_cat_fallback}, f, indent=2)



# Implementing the imputation for Item_price to create item_price_imputed

def price_imputer(row):
    if pd.notna(row['Item_price']):
        row['item_price_imputed'] = row['Item_price']
        return row
    else:
        cat = row['Product_category']
        if cat in medians_by_cat.keys():
            row['item_price_imputed'] = medians_by_cat_fallback[cat]
        else:
            row['item_price_imputed'] = medians_by_cat_fallback["__FALLBACK__"]
    return row


X_train=X_train.apply(price_imputer,axis = 1)

X_test=X_test.apply(price_imputer , axis = 1)


# %%
# item_price_log = np.log1p(data['item_price_imputed'])
# data['item_price_log'] = item_price_log
X_train['item_price_log'] = np.log1p(X_train['item_price_imputed'])
X_test['item_price_log'] = np.log1p(X_test['item_price_imputed'])


# %%
# Product Dummies



product_cat = sorted(X_train['Product_category'].unique())
ohe2 = OneHotEncoder(drop = 'first' , handle_unknown='ignore' , categories=[product_cat])
ohe2.fit(X_train[['Product_category']])
product_dummies_train = ohe2.transform(X_train[['Product_category']])
product_dummies_test = ohe2.transform(X_test[['Product_category']])


# Agent Shift Dummies

shift_cat = sorted(X_train['Agent Shift'].unique())
ohe3 = OneHotEncoder(drop = 'first' , handle_unknown='ignore' , categories=[shift_cat])
ohe3.fit(X_train[['Agent Shift']])
shift_dummies_train = ohe3.transform(X_train[['Agent Shift']])
shift_dummies_test = ohe3.transform(X_test[['Agent Shift']])



# saving these encoders
joblib.dump(ohe2, fr'{PREPROCESS_DIR}/ohe_product.joblib')
joblib.dump(ohe3, fr'{PREPROCESS_DIR}/ohe_shift.joblib')

# %%
cols_to_drop = ['channel_name', 'category', 'Sub-category', 'Customer Remarks',
       'order_date_time', 'Issue_reported at', 'issue_responded',
       'Survey_response_Date', 'Customer_City', 'Product_category',
       'Item_price', 'connected_handling_time', 'Tenure Bucket', 'Agent Shift','cleaned_remarks','order_date_time_imputed','time_to_issue','IssueResponseTime', 'price_category','handling_time_category','item_price_imputed','city_csat_mean']




# %%
# dropping these columns from X_train and X_test
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# saving the preprocessed data
X_train.to_csv(fr'data/processed/X_train_preprocessed.csv', index=False)
X_test.to_csv(fr'data/processed/X_test_preprocessed.csv', index=False)

# %%
# converting the X_train and X_test to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
# concatenating X_train_ohe1 and X_test_ohe1 with X_train and X_test respectively
X_train = np.concatenate((X_train, X_train_ohe1.toarray()), axis=1)
X_test = np.concatenate((X_test, X_test_ohe1.toarray()), axis=1)

# concatenating product_dummies_train and product_dummies_test with X_train and X_test respectively

X_train = np.concatenate((X_train, product_dummies_train.toarray()), axis=1)
X_test = np.concatenate((X_test, product_dummies_test.toarray()), axis=1)

# concatenating shift_dummies_train and shift_dummies_test with X_train and X_test respectively
X_train = np.concatenate((X_train, shift_dummies_train.toarray()), axis=1)
X_test = np.concatenate((X_test, shift_dummies_test.toarray()), axis=1)


# %%
# save these X_train and X_test as numpy arrays in ../data/processed/
# np.save('data/processed/X_train_preprocessed.npy', X_train)
# np.save('data/processed/X_test_preprocessed.npy', X_test)
np.save('data/processed/y_train.npy', y_train.to_numpy())
np.save('data/processed/y_test.npy', y_test.to_numpy())

print("Preprocessing pipeline script completed")
