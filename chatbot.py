import pickle
import nltk
import random
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nltk.download('punkt')

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([stemmer.stem(w) for w in tokens])

with open('model.pkl', 'rb') as f:
    model, vectorizer, intents = pickle.load(f)

def get_response(user_input):
    user_input = preprocess(user_input)
    X = vectorizer.transform([user_input])
    tag = model.predict(X)[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."
