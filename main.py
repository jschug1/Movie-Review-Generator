from functions2 import *
import json

########################################################################################
# The script which generates a randomly generated movie review based on imdb data.  Functions and data loaded from
# function2.py, data-gen.py, and next_word_dict_gen.py.
########################################################################################

f = open('next-word-vecs-3.json')
nwd = json.load(f)
nwd = normalize_vec(nwd)
f.close()

########################################################################################
# Generates a movie review with up to 10 sentences starting with 'This movie is'.
########################################################################################

sentence = sentence_chain(nwd, 'This movie is', 10)
print(sentence)
