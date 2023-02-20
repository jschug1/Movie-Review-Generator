This is a set of 4 files which generate a movie review based on the imdb movie review data set.  This is a niave algorithm used to practice coding and data manipulation.
The method consists of find all distinct k-word sequences, where k is some integer (The files data_gen.py and next_word_dict_gen.py use k = 3, but one can change k if one wishes),
and keeping track of which words appear after each k-word sequence and how many times it appears.  This is 'next word dictionary'.  The file data_gen.py makes a json file,
next-word-vecs-3.json', containing next word dictionaries for each reivew in the data set individually.
Then next_word_dict_gen.py combines all the small dictionaries into one big next word dictionary, stored in 'next-word-vecs-3'.

Then to make a movie review, start with a k-word sequence as a seed, then randomly choose the next word from the next word dictionary, move to the next 3 word sequence, and repeat until
you want to stop.  The current method is after some number of sentences, but can be easily amended if desired.  Again this is quite a niave algorithm I'm using to practice coding.
