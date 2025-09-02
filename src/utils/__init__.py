import pandas as pd
# import timedelta
from datetime import timedelta


def impute_order_datetime(row , avg_hours_diff):
    if pd.notna(row['order_date_time']):
        return row['order_date_time']
    elif pd.notna(row['Issue_reported at']):
        
        # Subtract average hours difference between order and issue reported
        hours_before = avg_hours_diff  # Use average hours difference calculated earlier
        return row['Issue_reported at'] - timedelta(hours=hours_before)



def categorize_price(price):
    if pd.isna(price):
        return 'Missing'
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
    

import re
import nltk
from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class CustomerRemarksPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Comprehensive text cleaning"""
        # Handle missing remarks token
        if text == '<Missing_Remark>':
            return text
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except important ones for sentiment
        # Keep some punctuation that might be important for sentiment
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text):
        """Tokenization and lemmatization"""
        if text == '<Unknown>':
            return [text]
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Filter out tokens that are just punctuation
        tokens = [token for token in tokens if not all(c in string.punctuation for c in token)]
        
        return tokens
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        return ' '.join(tokens)


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