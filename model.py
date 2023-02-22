from functions2 import *
import random

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def model(word_sequences, tokenizer, embedding_dim, lstm_units, learning_rate):

    total_words = len(tokenizer.word_index) + 1
    max_seq_len = max([len(x) for x in word_sequences])

    model = Sequential([
        Embedding(total_words, embedding_dim, input_length=max_seq_len - 1),
        Bidirectional(LSTM(lstm_units)),
        Dense(total_words, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model

def run_model(model, epochs, data, labels):

    history = model.fit(data, labels, epochs=epochs)

    return history