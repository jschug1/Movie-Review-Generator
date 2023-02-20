from functions2 import *
import tarfile
import json

########################################################################################
#This file generates a json file which contains a nested dictionary.  The keys will be the file names of the reviews.
#The first nested values will a dictionary with keys given by 3 word sequences from the review.
#The second values will be another dictionary with keys being the next words which follow the 3 word sequences, and values given by
# the number of times that follow up word appears.  For example, if the first review is
# "This movie is so good.  This movie is so good.  This movie is so bad."  Then the key "movie is so" will have value
# {"good": 2, "bad", 1}.
########################################################################################

########################################################################################
#First, we load the tar file containing all the imdb reviews.  I obtained this tarfile from https://github.com/ManavR123/ReviewRater.
#The list of names from the tarfile is stored as namelist
########################################################################################

file_name = 'aciimdb_v1.tar'
file_obj = tarfile.open(file_name)
namelist = list(file_obj.getnames())

########################################################################################
#Now initialize the dictionary as big_dict
########################################################################################

big_dict = {}

########################################################################################
# check if the file read is empty since namelist will contain directories divided into training and test sets.
# Otherwise,
# str1 = the string in the review file
# review1 = a pre-processed version of str1, see functions2 for documentation on pre-processing
# nwd1 = a dictionary of three word sequences in the keys and values containing next words data from the review
#       Here k = 3
########################################################################################

for name in namelist:
    file = file_obj.extractfile(name)
    if file != None:
        str1 = str(file.read())
        review1 = new_text_pre(str1)
        nwd1 = next_word_vectors(review1, 3)
        big_dict[name] = nwd1

########################################################################################
# Closes the tarfile and creates the json file
########################################################################################

file_obj.close()

json = json.dumps(big_dict)
f = open('next-word-vecs-2.json', 'w')
f.write(json)
f.close()