import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from preprocess import preprocess_text  # Import your preprocessing function

def train_model(df, model_name):
    # Handle missing values in 'Review text' column
    df['Review text'] = df['Review text'].fillna('')

    # Preprocess your data using preprocess_text from preprocess.py
    df['cleaned_text'] = df['Review text'].apply(preprocess_text)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    y = df['Ratings']  # Assuming 'Ratings' is your target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose the model based on input
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'SVM':
        model = SVC()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'Naive Bayes':
        model = MultinomialNB()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save trained model and vectorizer
    if not os.path.exists('models'):
        os.makedirs('models')

    model_filename = f'models/trained_model_{model_name.replace(" ", "_")}.pkl'
    joblib.dump(model, model_filename)
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

    return accuracy, report

# Run the training function if this script is executed directly
if __name__ == "__main__":
    df = pd.read_csv("data/sentiment.csv")  # Update with your dataset path
    model_names = ['Logistic Regression', 'SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Gradient Boosting', 'Decision Tree']
    for model_name in model_names:
        accuracy, report = train_model(df, model_name)
        print(f"{model_name} Model Accuracy: {accuracy:.2f}")
        print(f"{model_name} Classification Report:")
        print(report)
