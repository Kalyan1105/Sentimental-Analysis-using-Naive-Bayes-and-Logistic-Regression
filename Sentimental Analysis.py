import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
# Load and clean data
data = pd.read_csv('/content/IMDB_Dataset.csv')
# Function to remove HTML tags, URLs, and non-alphanumeric characters
def remove_tags(string):
    result = re.sub(r'<.*?>', '', string)  # Remove HTML tags
    result = re.sub(r'https?://\S+|www\.\S+', '', result)  # Remove URLs
    result = re.sub(r'[^a-zA-Z\s]', '', result)  # Remove non-alphanumeric characters
    return result.lower()

data['review'] = data['review'].apply(lambda cw: remove_tags(cw))
# Remove stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
# Lemmatize text
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st

data['review'] = data['review'].apply(lemmatize_text)
# Encode labels
reviews = data['review'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Split data into train and test sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    reviews, encoded_labels, stratify=encoded_labels, test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vec = TfidfVectorizer(max_features=5000)
X_train = vec.fit_transform(train_sentences)
X_test = vec.transform(test_sentences)
vocab = vec.get_feature_names_out()
# ------ Naive Bayes Implementation ------ #
X_train_array = X_train.toarray()
word_counts = {l: defaultdict(lambda: 0) for l in range(2)}
for i in range(X_train_array.shape[0]):
    l = train_labels[i]
    for j in range(len(vocab)):
        word_counts[l][vocab[j]] += X_train_array[i][j]
# Laplace smoothing for Naive Bayes
def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label, alpha=1):
    a = word_counts[text_label][word] + alpha
    b = n_label_items[text_label] + alpha * len(vocab)
    return math.log(a / b)

def group_by_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data

def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors

def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab:
                continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result

labels = [0, 1]
n_label_items, log_label_priors = fit(train_sentences, train_labels, labels)
pred_nb = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
# ------ Logistic Regression Implementation with Hyperparameter Tuning ------ #
logistic_model = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
grid_search.fit(X_train, train_labels)
best_logistic_model = grid_search.best_estimator_
pred_lr = best_logistic_model.predict(X_test)
# ------ Ensemble Approach ------ #
def ensemble_predict(pred_nb, pred_lr):
    ensemble_preds = []
    for nb, lr in zip(pred_nb, pred_lr):
        # Majority vote among the two classifiers
        ensemble_preds.append(int(round((nb + lr) / 2)))
    return ensemble_preds

ensemble_preds = ensemble_predict(pred_nb, pred_lr)
# ------ Model Evaluation ------ #
# Evaluate Naive Bayes
print("Naive Bayes Accuracy on test set:", accuracy_score(test_labels, pred_nb))
print("\nClassification Report for Naive Bayes:\n", classification_report(test_labels, pred_nb))
# Evaluate Logistic Regression
print("Logistic Regression Accuracy on test set:", accuracy_score(test_labels, pred_lr))
print("\nClassification Report for Logistic Regression:\n", classification_report(test_labels, pred_lr))
# Evaluate Ensemble Model
print("Ensemble Model Accuracy on test set:", accuracy_score(test_labels, ensemble_preds))
print("\nClassification Report for Ensemble Model:\n", classification_report(test_labels, ensemble_preds))
