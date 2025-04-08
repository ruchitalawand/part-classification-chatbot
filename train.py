import json
import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

with open('intents.json') as f:
    data = json.load(f)

corpus = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        labels.append(intent['tag'])

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([stemmer.stem(w) for w in tokens])

corpus = [preprocess(text) for text in corpus]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = labels

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer, data), f)
