from flask import Flask, request, jsonify
import random
import json
import re
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the model, vectorizer, and classes
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('classes.pkl', 'rb') as classes_file:
    classes = pickle.load(classes_file)

# Load intents data
try:
    with open('intents.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' file is missing.")
    exit(1)
except json.JSONDecodeError:
    print("Error: 'intents.json' file is not a valid JSON.")
    exit(1)

# Function to get the response based on the user message
def get_response(user_message):
    # Prepare the user message (lemmatizing, tokenizing)
    tokens = nltk.word_tokenize(user_message)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum()]
    input_data = vectorizer.transform([' '.join(tokens)])
    
    # Get the prediction from the model
    prediction = model.predict(input_data)
    predicted_class = prediction[0]
    
    # Find the response from the corresponding intent
    response = ""
    for intent in intents['intents']:
        if intent['tag'] == classes[predicted_class]:
            response = random.choice(intent['responses'])
            break
    if not response:
        response = "Sorry, I didn't understand that."
    
    return response

# Route to handle incoming messages via POST request
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    
    # Check if the message is empty
    if not user_message:
        return jsonify({"response": "Please send a message."}), 400
    
    # Get the bot response based on the user message
    response = get_response(user_message)
    
    return jsonify({"response": response})

# Route to handle the root URL (GET request)
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the chatbot API! Use the /chat endpoint for chat."})

if __name__ == "__main__":
    # Run the app in debug mode
    app.run(debug=True)
