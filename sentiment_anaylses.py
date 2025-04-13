!pip install gensim scikit-learn

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Libraries imported and setup complete.")

from google.colab import files

uploaded = files.upload()

data = pd.read_csv('amazon.csv')

data.head()


data = data[['reviewText', 'overall']].dropna()

def label_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

data['sentiment'] = data['overall'].apply(label_sentiment)

data.head()


# preprocessing function
def preprocess_text(text):

    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
data['cleaned_text'] = data['reviewText'].apply(preprocess_text)
data[['reviewText', 'cleaned_text']].head()


tokenized_reviews = [review.split() for review in data['cleaned_text']]

word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)

def get_vector(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(word_vectors) > 0:
        return sum(word_vectors) / len(word_vectors)
    else:
        return [0] * 100

data['vector'] = data['cleaned_text'].apply(get_vector)
print(data['vector'].head())


X = list(data['vector'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)


cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_clf = grid_search.best_estimator_
y_pred_tuned = best_clf.predict(X_test)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print("Tuned Accuracy:", accuracy_tuned)


results = grid_search.cv_results_

mean_test_scores = results['mean_test_score']
param_values = [str(params) for params in results['params']]

plt.figure(figsize=(10,6))
plt.plot(param_values, mean_test_scores, marker='o')
plt.xticks(rotation=90)
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Mean Test Score')
plt.title('Grid Search Results')
plt.show()
