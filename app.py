# app.py

import streamlit as st
import pandas as pd
from preprocess import preprocess_text
from train import train_model
from predict import predict_sentiment
import joblib

def load_data():
    try:
        df = pd.read_csv("data/sentiment.csv")  # Update with your dataset path
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")

def load_model(model_type):
    model_path = f'models/trained_model_{model_type}.pkl'  # Update with your model path
    vectorizer_path = 'models/tfidf_vectorizer.pkl'  # Update with your vectorizer path
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def main():
    st.title('Sentiment Analysis App')
    st.write('Enter your review text below:')
    
    df = load_data()
    df['cleaned_text'] = df['Review text'].fillna("").apply(preprocess_text)
    
    accuracy, report = train_model(df)
    
    review_text = st.text_area('Input Review Text:', height=200)
    
    model_type = st.selectbox('Select Model Type', 
                              ['Logistic_Regression', 'SVM', 'Random_Forest', 'Naive_Bayes', 
                               'KNN', 'Gradient_Boosting', 'Decision_Tree'])
    
    if st.button('Analyze Sentiment'):
        if review_text:
            model, vectorizer = load_model(model_type)
            preprocessed_text = preprocess_text(review_text)
            transformed_text = vectorizer.transform([preprocessed_text])
            predicted_sentiment = model.predict(transformed_text)[0]
            st.write(f'Predicted Sentiment: {predicted_sentiment}')
        else:
            st.warning('Please enter a review text.')

if __name__ == '__main__':
    main()
