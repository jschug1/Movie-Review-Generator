from functions2 import *
import tarfile
import json

file_name = 'aciimdb_v1.tar'
file_obj = tarfile.open(file_name)
namelist = list(file_obj.getnames())

nwd = {}

print('loaded')

i = 0
big_dict = {}
for name in namelist:
    file = file_obj.extractfile(name)
    if file != None:
        str1 = str(file.read())
        review1 = new_text_pre(str1)
        nwd1 = next_word_vectors(review1, 3)
        big_dict[name] = nwd1
    if i%1000 == 0:
        print(i)
    i += 1

print('dictionary')

file_obj.close()

json = json.dumps(big_dict)
f = open('next-word-vecs-2.json', 'w')
f.write(json)
f.close()