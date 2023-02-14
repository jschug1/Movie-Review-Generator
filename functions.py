import tarfile
from numpy import *
from numpy.random import choice

#################################################################################
# test
#################################################################################

def test1():
    print('test')



#################################################################################
# Open Tar File
#################################################################################


def load_namelist(file_name):

    file_obj_r = tarfile.open(file_name, "r")
    namelist = list(file_obj_r.getnames())
    file_obj_r.close()

    return namelist, file_obj_r

#################################################################################
# Incremental range
#################################################################################


def range_inc(start, stop, step, inc):
    i = start
    while i < stop:
        yield i
        i += step

#################################################################################
# Preprocessing the text files and making matrix
#################################################################################


def text_preprocess(sample_text):

    if sample_text[:2] == "b'" or 'b"':
        sample_text = sample_text[2:]
    if sample_text[len(sample_text)-1] == "'" or '"':
        sample_text = sample_text[:len(sample_text)-1]
    sample_text = sample_text.replace('<br />', '')
    sample_text = sample_text.replace("\\'", "'")
    sample_text = sample_text.replace('\n', '')
    for spaced in ['.', ',', '!', '?', '—', ':']:
        sample_text = sample_text.replace(spaced, ' {0} '.format(spaced))
    sample_text = sample_text.replace("(", "( ")
    sample_text = sample_text.replace(")", " ) ")
    sample_text = sample_text.replace("  ", " ")

    sample_text_words = sample_text.split(' ')
    if sample_text_words[len(sample_text_words)-1] == '':
        sample_text_words = sample_text_words[:len(sample_text_words)-1]

    return sample_text_words

#################################################################################
# Make big text file out of multiple files
#################################################################################


def combine_texts(file_obj, namelist):

    full_text = []

    for name in namelist:
        file = file_obj.extractfile(name)
        if file == None:
            continue
        sample_text_words = text_preprocess(str(file.read()))
        full_text += sample_text_words

    return full_text

#################################################################################
# Review Length
#################################################################################


def word_length(file_obj, namelist):

    rev_len_dict = {}

    for i, name in enumerate(namelist):
        file = file_obj.extractfile(name)
        if file == None:
            continue
        sample_text = str(file.read()).split(' ')
        rev_len_dict[i] = len(sample_text)

    return rev_len_dict

#################################################################################
# Making the word dictionary with order
#################################################################################


def dict_text(text):

    distinct_stw = list(set(text))
    word_indx_dict = {i: word for i, word in enumerate(distinct_stw)}

    return distinct_stw, word_indx_dict

#################################################################################
# Making dict of k sized word sets
#################################################################################


def k_set_word_list(text, k):

    sets_of_k_words = [' '.join(text[i:i+k])
                       for i, _ in enumerate(text[:-k+1])]
    distinct_k_word_sets = list(set(sets_of_k_words))

    return sets_of_k_words, distinct_k_word_sets

#################################################################################
# Making dictionary of all next words data for all distinct sets of k words
#################################################################################


def next_word_vectors(text, k):

    distinct_stw, word_indx_dict = dict_text(text)
    sets_of_k_words, distinct_k_word_sets = k_set_word_list(text, k)

    dict_temp = {}
    for i, word in enumerate(sets_of_k_words[:-1]):
        #print(i, word)
        # print(f'word_idx={word_idx}')
        next_word = text[i+k]
        # print(f'next_word_idx={next_word_idx}')
        if (word, next_word) in dict_temp.keys():
            dict_temp[(word, next_word)] += 1
        else:
            dict_temp[(word, next_word)] = 1

    next_word_dict = {}
    for tup in dict_temp.keys():
        if tup[0] in next_word_dict.keys():
            next_word_dict[tup[0]].append([tup[1], dict_temp[tup]])
        else:
            next_word_dict[tup[0]] = [[tup[1], dict_temp[tup]]]

    next_word_choices = {}

    for word_seq in next_word_dict.keys():
        s = 0
        word_choices = []
        probs = []
        tup = next_word_dict[word_seq]
        for pair in tup:
            s += pair[1]
            word_choices.append(pair[0])
            probs.append(pair[1])
        for i in range(len(probs)):
            probs[i] = probs[i]/s
        fakesum = sum(probs)
        probs[0] = probs[0] + (1 - fakesum)

        next_word_choices[word_seq] = [word_choices, probs]

    return next_word_choices

#################################################################################
# Defining next word function
#################################################################################


def next_word(next_word_choices, word_seq):
    word_seq_choices = next_word_choices[word_seq]
    [word_choices, probs] = word_seq_choices
    weighted_choice = choice(word_choices, 1, True, probs)
    return weighted_choice

#################################################################################
# Making next_word_vectors from filename
#################################################################################

def nvw_from_file(file_name, k):
    namelist, _ = load_namelist(file_name)
    file_obj_read = tarfile.open(file_name, 'r')
    full_text = combine_texts(file_obj_read, namelist)
    nvw = next_word_vectors(full_text, k)
    file_obj_read.close()

    return nvw

#################################################################################
# Making Fake Review functions
#################################################################################

# by number of words, may not end in period


def chain(next_word_vecs, k, seed, chain_length, seed_length):
    if len(seed.split(' ')) != seed_length:
        print('Not correct seed length')
        return None
    current_words = seed
    # print(current_words)
    sentence = seed
    for _ in range(chain_length):
        sentence += ' '
        nextword = next_word(next_word_vecs, current_words)
        sentence += nextword[0]
        dummy = current_words.split(' ')
        # print(dummy)
        dummy.append(nextword[0])
        # print(dummy)
        dummy = dummy[1:]
        # print(dummy)
        current_words = ' '.join(dummy)
        # print(current_words)
    return sentence

# by number of sentences, doesn't currently take into account ellipses


def sentence_chain(seed, sentence_num):
    current_words = seed
    # print(current_words)
    sentence = seed
    i = 0
    while i < sentence_num:
        sentence += ' '
        nextword = next_word(next_wocurrent_words)
        sentence += nextword[0]
        dummy = current_words.split(' ')
        # print(dummy)
        dummy.append(nextword[0])
        # print(dummy)
        dummy = dummy[1:]
        # print(dummy)
        current_words = ' '.join(dummy)
        # print(current_words)
        if nextword[0] == '.':
            i += 1

    return sentence
