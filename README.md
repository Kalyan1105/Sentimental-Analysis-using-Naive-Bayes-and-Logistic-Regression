# Sentimental-Analysis-using-Naive-Bayes-and-Logistic-Regression
Sentiment Analysis on IMDB Dataset
This project implements a sentiment analysis pipeline to classify movie reviews from the IMDB dataset as positive or negative. It combines text preprocessing, feature extraction, and machine learning models, including a custom Naive Bayes classifier, Logistic Regression with hyperparameter tuning, and an ensemble approach.
Overview
The code performs the following tasks:

Data Loading and Cleaning: Loads the IMDB dataset and cleans the text by removing HTML tags, URLs, and non-alphanumeric characters.
Text Preprocessing: Applies stop word removal and lemmatization using NLTK to standardize the text.
Feature Extraction: Converts text to numerical features using TF-IDF vectorization.
Model Training:
Implements a custom Naive Bayes classifier with Laplace smoothing.
Trains a Logistic Regression model with hyperparameter tuning using GridSearchCV.
Combines predictions from both models using an ensemble approach (majority voting).


Evaluation: Evaluates the performance of each model (Naive Bayes, Logistic Regression, and Ensemble) using accuracy and classification reports.

Requirements
To run this code, ensure you have the following dependencies installed:

Python 3.7+
Pandas
NumPy
Scikit-learn
NLTK
re (standard library)
math (standard library)

You can install the required packages using:
pip install pandas numpy scikit-learn nltk

Additionally, download the required NLTK resources by running the following in your Python environment:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

Input
The code expects a CSV file named IMDB_Dataset.csv with the following columns:

review: The text of the movie review.
sentiment: The sentiment label (positive or negative).

Update the file path if necessary:
data = pd.read_csv('/content/IMDB_Dataset.csv')  # Path to your CSV file

The dataset can be downloaded from Kaggle or other sources.
Usage

Prepare the Dataset: Ensure the IMDB_Dataset.csv file is accessible (e.g., uploaded to Google Colab or available locally).
Run the Script: Execute the Python script in a compatible environment (e.g., Google Colab, Jupyter Notebook, or local Python environment).
Output: The script will:
Preprocess the text data.
Train and evaluate three models: Naive Bayes, Logistic Regression, and an Ensemble model.
Print accuracy scores and classification reports (precision, recall, F1-score) for each model on the test set.



Key Parameters

TF-IDF Vectorizer:
max_features=5000: Limits the vocabulary to the top 5000 features.


Train-Test Split:
test_size=0.2: 20% of the data is used for testing.
random_state=42: Ensures reproducibility.


Logistic Regression Hyperparameter Tuning:
param_grid = {'C': [0.01, 0.1, 1, 10]}: Tests different regularization strengths.
cv=5: Uses 5-fold cross-validation.


Naive Bayes:
alpha=1: Laplace smoothing parameter to handle zero probabilities.



Models

Naive Bayes:
A custom implementation with Laplace smoothing.
Computes word probabilities for each class (positive/negative) based on TF-IDF features.


Logistic Regression:
Uses Scikit-learn's implementation with hyperparameter tuning for the regularization parameter C.
Trained on TF-IDF features.


Ensemble Model:
Combines predictions from Naive Bayes and Logistic Regression using majority voting.
Averages the predictions and rounds to the nearest integer (0 or 1).



Output
The script outputs:

Accuracy Scores: For Naive Bayes, Logistic Regression, and the Ensemble model on the test set.
Classification Reports: Detailed metrics (precision, recall, F1-score) for each model, broken down by class (0: negative, 1: positive).

Example output:
Naive Bayes Accuracy on test set: 0.85
Classification Report for Naive Bayes:
              precision    recall  f1-score   support
           0       0.84      0.86      0.85      5000
           1       0.86      0.84      0.85      5000
    accuracy                           0.85     10000
   macro avg       0.85      0.85      0.85     10000
weighted avg       0.85      0.85      0.85     10000

Logistic Regression Accuracy on test set: 0.88
...

Ensemble Model Accuracy on test set: 0.87
...

Notes

The dataset is split into 80% training and 20% testing, with stratification to maintain class balance.
The custom Naive Bayes implementation assumes binary classification (0 or 1).
The ensemble approach is a simple majority vote; more sophisticated ensemble methods could be explored.
Adjust max_features in the TF-IDF vectorizer or the param_grid for Logistic Regression to experiment with performance.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built using Scikit-learn for machine learning and NLTK for text preprocessing.
Dataset sourced from the IMDB movie review dataset.

