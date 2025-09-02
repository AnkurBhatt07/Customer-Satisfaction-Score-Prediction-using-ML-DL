import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings("ignore")


def your_data_loading_function():
    X_train = pd.read_csv('data/processed/train_val_test/X_train.csv')
    y_train = np.load('data/processed/train_val_test/y_train.npy')
    X_val = pd.read_csv('data/processed/train_val_test/X_val.csv')
    y_val = np.load('data/processed/train_val_test/y_val.npy')
    return X_train, y_train, X_val, y_val

def your_smote_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def your_smote_tomek_data(X_train, y_train):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    return X_resampled, y_resampled



# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_deep_learning_model(input_dim, num_classes=5, learning_rate=0.001, dropout_rate=0.3):
    """
    Create a deep neural network for CSAT multiclass classification
    """
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_dim=input_dim,),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # First hidden layer
        Dense(128, activation='relu',),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second hidden layer
        Dense(64, activation='relu',),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Third hidden layer
        Dense(32, activation='relu',),
        Dropout(dropout_rate),
        
        # Output layer with softmax for multiclass classification
        Dense(num_classes, activation='softmax',)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',  # For integer labels (0-4)
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    return model

def create_ordinal_aware_model(input_dim, num_classes=5, learning_rate=0.001, dropout_rate=0.3):
    """
    Alternative model that can be adapted for ordinal regression
    """
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        
        # For ordinal regression, you might want a single output with different activation
        Dense(num_classes, activation='softmax')  # Still using softmax for now
    ])
    
    # Custom loss for ordinal data could be implemented here
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_deep_learning_model(X_train, y_train, X_val, y_val, model_name="standard", 
                             epochs=100, batch_size=32, patience=10):
    """
    Train the deep learning model with callbacks
    """
    # Ensure labels are 0-based for multiclass classification
    y_train_encoded = y_train - 1 if y_train.min() == 1 else y_train
    y_val_encoded = y_val - 1 if y_val.min() == 1 else y_val
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_deep_learning_model(input_dim)
    
    # Print model summary
    print(f"\n=== {model_name.upper()} MODEL ARCHITECTURE ===")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n=== TRAINING {model_name.upper()} MODEL ===")
    history = model.fit(
        X_train, y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print(f"\n=== EVALUATING {model_name.upper()} MODEL ===")
    train_loss, train_acc, train_top_k = model.evaluate(X_train, y_train_encoded, verbose=0)
    val_loss, val_acc, val_top_k = model.evaluate(X_val, y_val_encoded, verbose=0)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Predictions for detailed evaluation
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Convert back to original labels (1-5) for reporting
    y_val_original = y_val_encoded + 1
    y_pred_original = y_pred_classes + 1
    
    # Classification report
    print(f"\n=== CLASSIFICATION REPORT - {model_name.upper()} ===")
    print(classification_report(y_val_original, y_pred_original, target_names=[f'CSAT_{i}' for i in range(1, 6)]))
    
    return model, history, y_pred_original

def plot_training_history(history, model_name):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'CSAT_{i}' for i in range(1, 6)],
                yticklabels=[f'CSAT_{i}' for i in range(1, 6)])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ===============================
# MAIN TRAINING PIPELINE
# ===============================

def main_training_pipeline():
    """
    Main function to train models on all three datasets
    """
    print("=== DEEP LEARNING TRAINING PIPELINE FOR CSAT CLASSIFICATION ===")
    
    # Dictionary to store results
    results = {}
    
    # Define your datasets (you'll need to load your actual data)
    datasets = {
        'original': (X_train, y_train, X_val, y_val),
        'smote': (X_train_smote, y_train_smote, X_val, y_val),
        'smote_tomek': (X_train_smote_tomek, y_train_smote_tomek, X_val, y_val)
    }
    
    # Train models on each dataset
    for dataset_name, (X_tr, y_tr, X_va, y_va) in datasets.items():
        print(f"\n{'='*60}")
        print(f"TRAINING ON {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Train model
        model, history, predictions = train_deep_learning_model(
            X_tr, y_tr, X_va, y_va, 
            model_name=dataset_name,
            epochs=100,
            batch_size=64,
            patience=15
        )
        
        # Plot training history
        plot_training_history(history, dataset_name)
        
        # Plot confusion matrix
        y_val_original = (y_va - 1 + 1) if y_va.min() == 1 else (y_va + 1)
        plot_confusion_matrix(y_val_original, predictions, dataset_name)
        
        # Store results
        results[dataset_name] = {
            'model': model,
            'history': history,
            'predictions': predictions,
            'val_accuracy': max(history.history['val_accuracy'])
        }
    
    # Compare results
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"{name.upper()} - Best Validation Accuracy: {result['val_accuracy']:.4f}")
    
    return results

# Advanced hyperparameter tuning function
def hyperparameter_search(X_train, y_train, X_val, y_val, n_trials=20):
    """
    Simple grid search for hyperparameters
    """
    import itertools
    
    # Define hyperparameter grid
    learning_rates = [0.001, 0.0001, 0.01]
    dropout_rates = [0.2, 0.3, 0.4]
    batch_sizes = [32, 64, 128]
    
    best_score = 0
    best_params = {}
    
    # Convert labels to 0-based
    y_train_encoded = y_train - 1 if y_train.min() == 1 else y_train
    y_val_encoded = y_val - 1 if y_val.min() == 1 else y_val
    
    for lr, dropout, batch_size in itertools.product(learning_rates, dropout_rates, batch_sizes):
        print(f"Testing: lr={lr}, dropout={dropout}, batch_size={batch_size}")
        
        # Create and train model
        model = create_deep_learning_model(
            X_train.shape[1], 
            learning_rate=lr, 
            dropout_rate=dropout
        )
        
        # Train with early stopping
        history = model.fit(
            X_train, y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=50,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        # Get best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        
        if best_val_acc > best_score:
            best_score = best_val_acc
            best_params = {
                'learning_rate': lr,
                'dropout_rate': dropout,
                'batch_size': batch_size
            }
        
        print(f"Validation Accuracy: {best_val_acc:.4f}")
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_score:.4f}")
    
    return best_params

# Example usage:
if __name__ == "__main__":
    # Load your preprocessed data
    X_train, y_train, X_val, y_val = your_data_loading_function()
    X_train_smote, y_train_smote = your_smote_data(X_train , y_train)
    X_train_smote_tomek, y_train_smote_tomek = your_smote_tomek_data(X_train , y_train)

    # Run the main training pipeline
    results = main_training_pipeline()
    
    # Optional: Run hyperparameter search on best performing dataset
    # best_params = hyperparameter_search(X_train_smote, y_train_smote, X_val, y_val)
    
    pass

