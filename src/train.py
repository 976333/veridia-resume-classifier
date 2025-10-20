import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model(X_train, y_train, X_test, y_test, model_path='model/classifier.pkl'):
    """Train a classifier and evaluate performance."""
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Initialize and train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train_encoded)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
    
    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save model and encoder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    print(f"Model saved to {model_path}")
    print("Label encoder saved to model/label_encoder.pkl")
    
    return model, label_encoder

if __name__ == "__main__":
    from preprocess import load_and_preprocess_data, create_features
    
    # Load and preprocess
    df = load_and_preprocess_data("data/resume_dataset.csv")
    X, y, vectorizer = create_features(df, save_vectorizer=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model, label_encoder = train_model(X_train, y_train, X_test, y_test)