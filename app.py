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

def load_model(model_name):
    model_path = f'models/trained_model_{model_name.replace(" ", "_")}.pkl'  # Update with your model path
    vectorizer_path = 'models/tfidf_vectorizer.pkl'  # Update with your vectorizer path
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def main():
    st.title('Sentiment Analysis App')
    st.write('Enter your review text below:')
    
    df = load_data()
    df['cleaned_text'] = df['Review text'].fillna("").apply(preprocess_text)
    
    classifier_options = ['Logistic Regression', 'SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Gradient Boosting', 'Decision Tree']
    classifier_choice = st.selectbox('Select Classifier', classifier_options)

    accuracy, report = train_model(df, classifier_choice)
    # st.write(f"{classifier_choice} Model Accuracy: {accuracy:.2f}")
    # st.write(f"{classifier_choice} Classification Report:")
    # st.text(report)
    
    review_text = st.text_area('Input Review Text:', height=200)
    
    if st.button('Analyze Sentiment'):
        if review_text:
            model, vectorizer = load_model(classifier_choice)
            preprocessed_text = preprocess_text(review_text)
            transformed_text = vectorizer.transform([preprocessed_text])
            predicted_sentiment = model.predict(transformed_text)[0]
            st.write(f' Ratings range 1-5 \n Predicted Sentiment: {predicted_sentiment} ratings')
        else:
            st.warning('Please enter a review text.')

if __name__ == '__main__':
    main()
