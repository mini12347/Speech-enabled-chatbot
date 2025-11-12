import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
import string
import random
import json

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
with open('./data.json','rb') as f:
    intents_json = json.load(f)

def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

intents_data = {}
for intent in intents_json.get('intents', []):
    tag = intent.get('tag', '')
    patterns = intent.get('patterns', [])
    responses = intent.get('responses', [])
    if tag and patterns and responses:
        intents_data[tag] = {
            'patterns': [preprocess(p) for p in patterns],
            'responses': responses
        }

def find_best_match(user_input):
    if not user_input or not user_input.strip():
        return "no-response", None
    
    user_tokens = preprocess(user_input)
    if not user_tokens:
        return "no-response", None
    
    max_similarity = 0
    best_intent = "default"
    best_response = None
    
    for tag, intent_data in intents_data.items():
        for pattern_tokens in intent_data['patterns']:
            if not pattern_tokens:
                continue
            union = set(user_tokens).union(pattern_tokens)
            if len(union) > 0:
                similarity = len(set(user_tokens).intersection(pattern_tokens)) / float(len(union))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_intent = tag
                    best_response = random.choice(intent_data['responses'])
    
    if max_similarity > 0.1:
        return best_intent, best_response
    
    return "default", random.choice(intents_data.get("default", {}).get("responses", ["I'm here to support you. Could you tell me more about how you're feeling?"]))

def transcribe_audio():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source, timeout=10)
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Error with speech recognition service"
    except Exception as e:
        return f"Error: {str(e)}"

def chatbot(question):
    intent, response = find_best_match(question)
    return response if response else "I'm here to support you. Could you tell me more about how you're feeling?"
def app(): 
    st.set_page_config(page_title="Speech Chatbot", layout="centered")
    st.title("Speech-Enabled Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎤 Record Audio"):
            user_input = transcribe_audio()
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                response = chatbot(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        user_text = st.text_input("Or type your message:")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            user_text.text=''
            response = chatbot(user_text)
            st.session_state.messages.append({"role": "assistant", "content": response})
if __name__=="__main__":

    app()
