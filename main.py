from functions2 import *
from model import *
import tarfile
import json
import random

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

########################################################################################
# The script which generates a randomly generated movie review based on imdb data.  Functions and data loaded from
# function2.py, data-gen.py, and next_word_dict_gen.py.
########################################################################################

#f = open('next-word-vecs-3.json')
#nwd = json.load(f)
#nwd = normalize_vec(nwd)
#f.close()

########################################################################################
# Generates a movie review with up to 10 sentences starting with 'This movie is'.
########################################################################################

#sentence = sentence_chain(nwd, 'This movie is', 10)
#print(sentence)

########################################################################################
# New feature for this commit.  The following generates a review based on a simple machine learning algorthm.
# A few functions from functions2.py and model loaded from model.py
########################################################################################

filename = 'aciimdb_v1.tar'
file_obj = tarfile.open(filename, 'r')
namelist = list(file_obj.getnames())
namelist = namelist[20:30] # or however many files you want to obtain data from

corpus = combinetexts(namelist, file_obj)
corpus = corpus.split('.') # split the reviews up by sentences, but could use another split, perhaps by review
file_obj.close()

tokenizer = ml_tokenizer(corpus)
word_sequences = word_seq(corpus, tokenizer) # initialize the tokenizer and create word sequences

data, labels = make_data(word_sequences, tokenizer) # make the data and labels

embedding_dim = 16
lstm_units = 64
learning_rate = 0.001
epochs = 120 # you can set some parameters for the model here

gen_model = model(word_sequences, tokenizer, embedding_dim, lstm_units, learning_rate)
history = run_model(gen_model, epochs, data, labels) # run the model and fit the data

review = ml_movie_gen('This movie is so', next_words=50, model=gen_model, tokenizer=tokenizer, word_sequences=word_sequences)
print(review) # print the generated review





