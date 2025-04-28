import json
import random
import pickle



from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Initialize
lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare data
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Training data
training_sentences = []
training_labels = []

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0] if w.isalnum()]
    sentence = " ".join(pattern_words)
    training_sentences.append(sentence)
    training_labels.append(classes.index(doc[1]))

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)
y = training_labels

# Model
model = MultinomialNB()
model.fit(X, y)

# Save model and data
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("✅ Chatbot model trained and saved!")
