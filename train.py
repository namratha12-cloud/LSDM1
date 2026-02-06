import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_preprocess_data
from model import build_lstm_model
import matplotlib.pyplot as plt

def train_model(filepath):
    # Load and preprocess
    print("Loading and preprocessing data...")
    X, y, classes, tokenizer = load_and_preprocess_data(filepath)
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build model
    print(f"Building LSTM model for {num_classes} classes...")
    model = build_lstm_model(vocab_size, num_classes)
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=classes))
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('resume_lstm_model.h5')
    print("Model saved to resume_lstm_model.h5")
    
    return history, model

if __name__ == "__main__":
    train_model(r'C:\LSTM\Resume\Resume.csv')
