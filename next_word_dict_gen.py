from functions2 import *
import tarfile
import json
import random

f = open('next-word-vecs-2.json')
nwd = json.load(f)

j = 0
nw_dict = {}
for key in nwd.keys():
    temp_dict = nwd[key]
    nw_dict = update_nwd(temp_dict, nw_dict)
    if j%1000 == 0:
        print(j)
    j += 1

f.close()

print(len(nw_dict))

json = json.dumps(nw_dict)
f = open('next-word-vecs-3.json', 'w')
f.write(json)
f.close()

