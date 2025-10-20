import joblib
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import os

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """Clean and normalize resume text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(words)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def predict_category_from_text(resume_text):
    """Predict category from raw resume text."""
    model = joblib.load('model/classifier.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    
    cleaned = clean_text(resume_text)
    if not cleaned.strip():
        raise ValueError("No valid text found in resume.")
    
    features = vectorizer.transform([cleaned])
    pred_encoded = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    category = label_encoder.inverse_transform([pred_encoded])[0]
    
    return {
        'category': category,
        'confidence': float(confidence),
        'all_probabilities': dict(zip(label_encoder.classes_, model.predict_proba(features)[0]))
    }

def predict_category(file_path):
    """Unified function: accepts .txt or .pdf file path."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif file_ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload .txt or .pdf")
    
    return predict_category_from_text(text)