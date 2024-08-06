import joblib
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(model_path, vectorizer_path):
    # Load trained model and TF-IDF vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(review_text, model_name):
    # Define paths to your trained model and vectorizer
    model_path = f'models/trained_model_{model_name.replace(" ", "_")}.pkl'  # Update with your model path
    vectorizer_path = 'models/tfidf_vectorizer.pkl'  # Update with your vectorizer path
    
    # Load the model and vectorizer
    model, vectorizer = load_model(model_path, vectorizer_path)
    
    # Preprocess text
    cleaned_text = preprocess_text(review_text)
    
    # Transform text using TF-IDF vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict sentiment
    sentiment = model.predict(text_vectorized)[0]
    
    return sentiment
