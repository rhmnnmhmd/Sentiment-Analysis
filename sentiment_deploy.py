import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, Masking, Bidirectional, Embedding
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import matplotlib.pyplot as plt

import os

import datetime as dt

import seaborn as sns

import pandas as pd

import sklearn 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np

import re

import json

import pickle


def conv(x):
    my_dict = {0: 'negative',
               1: 'positive'}
               
    return my_dict[x]

conv = np.vectorize(conv)

# load model
MODEL_PATH = os.path.join(os.getcwd(), "sentiment_model.h5")
loaded_model = load_model(MODEL_PATH)

# load tokenizer
TOKENIZER_PATH = os.path.join(os.getcwd(), "tokenizer.json")
with open(TOKENIZER_PATH, "r") as file:
    loaded_tokenizer = json.load(file)
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)

# load OHE
OHE_PATH = os.path.join(os.getcwd(), "ohe.pkl")
with open(OHE_PATH, "rb") as file:
    loaded_ohe = pickle.load(file)

user_review = input('Type your review: ')

input_review = pd.Series([user_review], name='review')
input_review = input_review.str.replace('<.*?>', ' ')
input_review = input_review.str.replace('[^a-zA-Z]', ' ')
input_review = input_review.str.lower()

input_review_tokenized = loaded_tokenizer.texts_to_sequences(input_review)

max_len = 180
input_review_tokenized_padded = pad_sequences(input_review_tokenized, maxlen=max_len, padding='post', truncating='post')
# input_review_tokenized_padded.reshape(180, 1)

predicted_sentiment = loaded_model.predict(input_review_tokenized_padded)

predicted_sentiment_decoded = np.argmax(predicted_sentiment, axis=1)

predicted_sentiment_decoded = conv(predicted_sentiment_decoded).astype(object)

print(f'Your review have {predicted_sentiment_decoded[0]} sentiment')