import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """Clean and normalize resume text."""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(words)

def load_and_preprocess_data(data_path):
    """Load dataset and preprocess text."""
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Drop rows with missing resume text or category
    df.dropna(subset=['Resume_str'], inplace=True)
    df.dropna(subset=['Category'], inplace=True)
    
    # Clean resume text
    df['cleaned_resume'] = df['Resume_str'].apply(clean_text)
    
    # Optional: You can add more preprocessing steps like lemmatization here
    
    print(f"Dataset after cleaning: {df.shape}")
    return df

def create_features(df, save_vectorizer=True, vectorizer_path='model/vectorizer.pkl'):
    """Convert text to TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_resume'])
    y = df['Category']
    
    if save_vectorizer:
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")
    
    return X, y, vectorizer

if __name__ == "__main__":
    # Example usage
    data_path = "data/resume_dataset.csv"  # Update path as needed
    df = load_and_preprocess_data(data_path)
    X, y, vectorizer = create_features(df)
    print("Preprocessing completed.")