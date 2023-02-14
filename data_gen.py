from functions import *
import json
from numpy.random import choice

file_name = 'aciimdb_v1.tar'
namelist, _ = load_namelist(file_name)
file_obj = tarfile.open(file_name, 'r')

fulltext = combine_texts(file_obj, namelist)

nwv = next_word_vectors(fulltext, 3)
file_obj.close()

json = json.dumps(nwv)
f = open('next-word-vecs.json', 'w')
f.write(json)
f.close()