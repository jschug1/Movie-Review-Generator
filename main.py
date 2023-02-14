from functions import *
import json

f = open('next-word-vecs.json')
nw_dict = json.load(f)
sentence = chain(nw_dict, 3, 'This movie is', 150, 3)
print(sentence)

f.close()

