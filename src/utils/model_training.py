import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Add, Layer
from tensorflow.keras.optimizers import Adam
import os

# Fix random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load your processed and scaled data
X_train = pd.read_csv('X_train_transformed.csv')
X_test = pd.read_csv('X_test_transformed.csv')
y_train = pd.read_csv('y_train.csv')['target']
y_test = pd.read_csv('y_test.csv')['target']

# 1. Define evaluation function
def evaluate_model(y_true, y_pred, name='Model'):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Results:\n MAE: {mae:.3f} \n RMSE: {rmse:.3f} \n R2: {r2:.3f}\n")
    return mae, rmse, r2

# 2. Try classical ML models
def train_linear_regression():
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    joblib.dump(model, 'models/linear_regression_model.joblib')
    evaluate_model(y_test, pred, 'Linear Regression')

def train_ridge():
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    joblib.dump(model, 'models/ridge_model.joblib')
    evaluate_model(y_test, pred, 'Ridge Regression')

def train_random_forest():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    joblib.dump(model, 'models/rf_model.joblib')
    evaluate_model(y_test, pred, 'Random Forest')

def train_xgboost():
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    joblib.dump(model, 'models/xgb_model.joblib')
    evaluate_model(y_test, pred, 'XGBoost')

# 3. Define Deep Neural Network (Keras)
def build_keras_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',  # MAE or MSE
                  metrics=['mae'])
    return model

def train_keras():
    model = build_keras_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                        validation_split=0.2, verbose=1,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                        ])
    model.save('models/keras_deep_model.h5')
    # Predict
    y_pred = model.predict(X_test).flatten()
    # Round and clip for CSAT scores
    y_pred_int = np.clip(np.round(y_pred), 1, 5).astype(int)
    evaluate_model(y_test, y_pred_int, 'Deep Learning Model (Keras)')

# 4. Call all models

models_dir = os.path.join("artifacts", "models")
os.makedirs(models_dir, exist_ok=True)

print("Training Linear Regression...")
train_linear_regression()
print("Training Ridge Regression...")
train_ridge()
print("Training Random Forest...")
train_random_forest()
print("Training XGBoost...")
train_xgboost()
print("Training Deep Neural Network...")
train_keras()

print("All models trained and evaluated.")


