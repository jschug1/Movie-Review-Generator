from functions2 import *
import json

f = open('next-word-vecs-3.json')
nwd = json.load(f)
nwd = normalize_vec(nwd)
f.close()

sentence = sentence_chain(nwd, 'This movie is', 10)
print(sentence)
