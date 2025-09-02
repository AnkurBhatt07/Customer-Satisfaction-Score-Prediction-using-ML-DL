CSAT Score Prediction: Comprehensive Analysis & Preprocessing Strategy

Executive Summary

Based on my analysis of the eCommerce customer support dataset, this report provides a comprehensive examination of 85,907 customer support interactions for CSAT score prediction. The analysis reveals significant challenges including severe class imbalance (69.4% of customers rate satisfaction as 5/5) and extensive missing data across multiple features, requiring sophisticated preprocessing strategies.


Dataset Overview & Business Context
The dataset contains 85,907 customer support interactions spanning from January to December 2023, with the primary objective of predicting Customer Satisfaction (CSAT) scores on a 1-5 scale. This predictive capability is crucial for improving customer experience and operational efficiency in eCommerce support operations.

Key Dataset Characteristics:

Total Records: 85,907 customer support interactions

Original Features: 20 variables covering channels, categories, timestamps, and customer feedback

Target Variable: CSAT Score (1-5 scale)

Business Impact: Model accuracy and interpretability for actionable insights

Critical Data Quality Issues
Missing Data Analysis

The dataset exhibits substantial missing data challenges that require strategic handling:

connected_handling_time: 99.72% missing (85,665 records) - Recommend complete removal

Customer_City: 80.12% missing (68,828 records) - Geographic information loss

Product_category: 79.98% missing (68,711 records) - Product context missing

Item_price: 79.97% missing (68,701 records) - Financial context unavailable

order_date_time: 79.96% missing (68,693 records) - Temporal analysis limited

Customer Remarks: 66.54% missing (57,165 records) - Critical text data gaps


Target Variable Characteristics
The CSAT score distribution reveals severe class imbalance:

Score 5: 59,617 records (69.40%) - Highest satisfaction

Score 4: 11,219 records (13.06%)

Score 1: 11,230 records (13.07%) - Lowest satisfaction

Score 3: 2,558 records (2.98%)

Score 2: 1,283 records (1.49%) - Least common

This extreme imbalance toward high satisfaction scores presents significant modeling challenges and suggests potential response bias or data collection issues.


Feature Importance & Statistical Relationships
Strong Predictive Features
Sentiment Analysis emerges as the most powerful predictor with a correlation of 0.522 between customer remark sentiment and CSAT scores. This finding indicates that natural language processing of customer feedback provides substantial predictive value.

Channel Performance Analysis shows statistically significant differences (Chi-square p-value < 0.001):

Email: 3,023 interactions (3.5%), mean CSAT: 3.90 - Lowest satisfaction

Inbound: 68,142 interactions (79.3%), mean CSAT: 4.25 - Primary channel

Outcall: 14,742 interactions (17.2%), mean CSAT: 4.27 - Highest satisfaction

Category Impact demonstrates significant variation across support categories (ANOVA p-value < 0.001), with returns and order-related issues comprising the majority of interactions.

Response Time Correlation shows a negative relationship (-0.089) between issue response time and customer satisfaction, though the effect is modest.


Comprehensive Preprocessing Strategy
Phase 1: Data Cleaning & Quality Enhancement
Feature Elimination Strategy:

Remove features with >95% missing data (connected_handling_time)

Drop identifier columns (Unique id, Order_id) that provide no predictive value

Consolidate low-frequency categorical levels to reduce dimensionality

Strategic Missing Value Treatment:

Customer Remarks (66.54% missing):

Implement placeholder strategy with <Missing_Remark> indicator

Create binary feature has_missing_remarks for pattern recognition

Apply advanced NLP preprocessing for non-missing text

Geographic Data (Customer_City - 80.12% missing):

Replace missing values with Unknown category

Create city_category feature based on mean CSAT scores (1-5 buckets)

Maintain geographic signal while managing sparsity

Temporal Data (order_date_time - 79.96% missing):

Implement distribution-based imputation for available dates

Extract temporal features: season, time_period, business_hours

Create interaction features between time and channel



Phase 2: Advanced Feature Engineering
Categorical Encoding Strategy:

High-cardinality category handling
Sub-category reduction: 57 → 27 categories by combining <300 record groups
Apply regularized target encoding with K-fold validation to prevent overfitting
Frequency encoding for category importance weighting

Text Analytics Pipeline:

Multi-level text feature extraction:
1. Basic statistics: character length, word count, sentence count
2. Sentiment analysis: polarity (-1 to 1) and subjectivity (0 to 1) 
3. Advanced NLP: TF-IDF vectorization, topic modeling with LDA
4. Deep learning: LSTM-based text embeddings for semantic understanding

Temporal Feature Engineering for analysis purpose:

Business-context temporal features:
Business hours indicator (9 AM - 5 PM)
Weekend/weekday classification  
Seasonal patterns (Winter, Spring, Summer, Fall)
Time period categorization (Morning, Afternoon, Evening, Night)

