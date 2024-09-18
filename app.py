import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download("stopwords")

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set a background color using Streamlit's markdown
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stTextArea, .stButton {
        background-color: #f7f9fc;
        color: #0f4c75;
        border: 2px solid #3282b8;
        border-radius: 10px;
        padding: 10px;
    }
    .stTextArea textarea, .stButton button {
        color: #0f4c75;
        font-size: 16px;
    }
    .stButton button {
        background-color: #3282b8;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stHeader {
        color: #0f4c75;
    }
    .title {
        color: #0f4c75;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 1.1em;
        color: #0f4c75;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add title and description
st.markdown("<h1 class='title'>Email / Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='description'>This tool helps you detect whether a message is spam or not. Simply enter the text of an email or message below, and we will analyze it for you!</p>",
    unsafe_allow_html=True,
)

# Get input from the user with a styled text area
input_sms = st.text_area('Enter the message', height=150, max_chars=500, help="Type or paste your email message here")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # Tokenize the text
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Add a predict button and style it
if st.button("Predict"):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        # Display the result with a nice header
        if result == 1:
            st.markdown("<h2 style='color: red; text-align: center;'>⚠️ Spam Alert! ⚠️</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green; text-align: center;'>✅ This is Not Spam ✅</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to classify.")
        
# Add some footer or note
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; font-size: 0.9em; color: #0f4c75;'>
        Created with ❤️ by Aryan 
    </div>
    """,
    unsafe_allow_html=True
)

