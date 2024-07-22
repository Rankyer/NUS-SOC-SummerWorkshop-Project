import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize recognizer
r = sr.Recognizer()

# Function to recognize speech from audio file or microphone
def recognize_speech_from_mic():
    with sr.Microphone() as source:
        print("Please wait. Calibrating microphone...")
        r.adjust_for_ambient_noise(source, duration=5)
        print("Microphone calibrated. Say something!")
        
        audio = r.listen(source, phrase_time_limit=5)
        
        try:
            print("Recognizing...")
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")



# Define the function to extract keywords by category
def extract_keywords_by_category(text, categories):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    categorized_keywords = {category: [] for category in categories}
    for category, keywords in categories.items():
        categorized_keywords[category] = [word for word in filtered_words if word in keywords]
    return categorized_keywords

# Define categories and their keywords
categories = {
    'animals': ['cat', 'dog', 'bird'],
    'food': ['apple', 'banana', 'watermelon']
}

# Capture speech and extract keywords by category
speech_text = recognize_speech_from_mic()
if speech_text:
    categorized_keywords = extract_keywords_by_category(speech_text, categories)
    for category, keywords in categorized_keywords.items():
        print(f"{category.capitalize()} Keywords: {keywords}")
