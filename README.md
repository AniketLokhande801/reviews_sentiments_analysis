# Flipkart Product Review Sentiment Analysis

This repository contains a project that performs sentiment analysis on Flipkart product reviews. The goal is to classify reviews using a machine learning model. The project includes text preprocessing, TF-IDF vectorization, model training, and prediction.

## Project Overview

The aim of this project is to analyze customer reviews on Flipkart and classify them into sentiment categories. The project achieves a  accuracy of 94% using various machine learning models.

### Key Features:
- **Sentiment Analysis Model:** Developed to classify product reviews as positive or negative.
- **Text Preprocessing:** Implemented techniques such as tokenization, stop-word removal, and stemming.
- **TF-IDF Vectorization:** Used to convert text data into numerical features for model training.
- **Accuracy:** Achieved 94% accuracy with machine learning models.

## Project Structure

- `app.py`: Main application file for running the sentiment analysis model as a web app.
- `predict.py`: Script for making predictions on new reviews.
- `preprocess.py`: Script containing functions for text preprocessing.
- `train.py`: Script for training the machine learning models on the dataset.
- `reviews.csv`: CSV file containing the dataset used for model training.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AniketLokhande801/Flipkart_Product_Review_Sentiment_Analysis.git
    cd Flipkart_Product_Review_Sentiment_Analysis
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Running the Application:**

    To start the web application, run the following command:
    ```bash
    streamlit run app.py
    ```

2. **Predicting Sentiment:**

    - Use the web interface to input a product review and predict its sentiment.
    - You can also use the `predict.py` script to predict sentiments for multiple reviews in batch mode.

3. **Training the Model:**

    To train the sentiment analysis model, run the `train.py` script:
    ```bash
    python train.py
    ```

## Contributing

Contributions are welcome! If you have suggestions for improvements, please open an issue or submit a pull request.

## Contact

For any inquiries, please contact Aniket Somnath Lokhande at aniketlokhande3654@gmail.com.
