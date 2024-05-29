from nltk.stem import WordNetLemmatizer
import spacy
import nltk
from nltk.corpus import stopwords
import string
import re
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

import contractions
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

app = Flask(__name__)

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laughter",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "LOL": "Laughing out loud",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "IDC": "I don’t care",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "LMAO": "Laughing my a** off",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing",
}


def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


stop_words = stopwords.words('english')


def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text


lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    word_list = word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output


def do_tokenization(text):

    token_words = word_tokenize(text)
    return token_words


tokenizer = Tokenizer(num_words=5000, split=' ')


# Define your text preprocessing functions
def remove_html_tags(text):
    return re.sub('<.*?>', '', text)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def stopwords_removal(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii')

# def expand_contractions(text):
#     # Implement your contractions expansion logic here
#     return text


@app.route('/')
def index():
    return render_template('index.html')


model = load_model('./RNN_MOVIE_REVIEW.h5')


@app.route('/predict', methods=['POST'])
def predicate():
    input_text = request.form['review']
    print(input_text)
    # change lower case
    input_text = input_text.lower()
    # remove html tags
    input_text = remove_html_tags(input_text)
    # remove punctuation
    input_text = remove_punctuation(input_text)
    # chat conversion
    input_text = chat_conversion(input_text)
    # remove stopwords
    input_text = stopwords_removal(input_text)
    # remove emoji
    input_text = remove_emoji(input_text)
    # expand contractions
    input_text = expand_contractions(input_text)
    # lemmatization
    input_text = lemmatize_text(input_text)
    # tokenization
    input_text = do_tokenization(input_text)
    tokenizer.fit_on_texts(input_text)
    input_text = tokenizer.texts_to_sequences(input_text)
    input_text = pad_sequences(input_text, maxlen=250)
    prediction = model.predict(input_text)
    print(prediction)
    if prediction > 0.5:
        return 'Positive Review'
    else:
        return 'Negative Review'
