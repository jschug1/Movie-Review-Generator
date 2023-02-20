from numpy.random import choice

########################################################################################
# This file contains functions used to make the niave movie review generator.
########################################################################################

########################################################################################
# function dict_text: returns list of distinct words from text as well as indexed dictionary

# input: text in the form a list split by ' '

# output: distinct_stw = list of distinct words from the text
# output: word_indx_dict = dictionary with keys given by integers and values given by associated word
########################################################################################

def dict_text(text):

    distinct_stw = list(set(text))
    word_indx_dict = {i: word for i, word in enumerate(distinct_stw)}

    return distinct_stw, word_indx_dict

########################################################################################
# function k_set_word_list: returns list of sets of k words in a row from text, as well as an indexed dictionary

# input: text in the form of a list of words split by ' '
# input: k, an integer, the number of words you want in a row, should be less than the size of the text

# output: sets_of_k_words = list of k word sets from text
# output: distinct_k_word_sets = list of distinct k word sets from the text
########################################################################################

def k_set_word_list(text, k):

    sets_of_k_words = [' '.join(text[i:i+k])
                       for i, _ in enumerate(text[:-k+1])]
    distinct_k_word_sets = list(set(sets_of_k_words))

    return sets_of_k_words, distinct_k_word_sets

########################################################################################
# function next_word_vectors: returns a dictionary with the keys being the k word sets
# and the values are a dictionary of the next words which appear, with values the number of times
# that next word appears

# input: text in the form a list of words split by ' '
# input: k, an integer, number of words you want in a row

# output: dict_temp = dictionary with keys given by distinct k word sets and values given by a
#                       a dictionary of next words with values the number of time that next word appears

# Example:  text = 'This movie is so good.  This movie is so bad.  This movie is so bad.'.split(' ')
#           k = 3
#           #dict_temp['movis is so'] = {'good': 1, 'bad': 2}
########################################################################################

def next_word_vectors(text, k):

    distinct_stw, word_indx_dict = dict_text(text)
    sets_of_k_words, distinct_k_word_sets = k_set_word_list(text, k)

    dict_temp = {}
    for i, word in enumerate(sets_of_k_words[:-1]):
        #print(i, word)
        # print(f'word_idx={word_idx}')
        next_word = text[i+k]
        # print(f'next_word_idx={next_word_idx}')
        if word not in dict_temp.keys():
            dict_temp[word] = {next_word: 1}
        else:
            if next_word not in dict_temp[word].keys():
                (dict_temp[word])[next_word] = 1
            else:
                (dict_temp[word])[next_word] += 1

    return dict_temp

########################################################################################
# function normalize_vec: takes a dictionary with data on how many times a next word appears
# and normalizes the values to be probabilities of the next word appearing

# input: dict = dictionary of next words appearing after a k word set as in 'next_word_vectors'

# output: dict = editing dictionary in the same format but values changed from number of times
#                   word appears to probability of appearing next
########################################################################################

def normalize_vec(dict):
    for word in dict.keys():
        sum = 0
        for next_word in dict[word].keys():
            sum += (dict[word])[next_word]
        for next_word in dict[word].keys():
            (dict[word])[next_word] = float((dict[word])[next_word])/float(sum)

    return dict

########################################################################################
# function new_text_pre: basic pre-processing for the text to be usable by the other functions
# It does the following:
#   - puts a token "<END>" at the end of each review to keep track of when each review ends
#   - removes the string b" at the beginning of each review
#   - gets rid of quotation mark the end of the review
#   - some html formatting appears periodically in the data set, removes some html format seen
#   - removes extra slashes or apostrophes for simplicity
#   - removes next line formatting
#   - removes isolated punctuation from simplicity
#   - removes parentheses for simplicity
#   - replaces extra spacing for formatting purposes
#   - then splits the text by ' ' making a list of each word

# input: sample text in the form a string

# output: list of words in text reformatted to fit other functions
########################################################################################

def new_text_pre(sample_text):
    sample_text += ' <END>'
    if sample_text[:2] == "b'" or sample_text[:2] == 'b"':
        sample_text = sample_text[2:]
    if sample_text[-1] == "'" or sample_text[-1] == '"':
        sample_text = sample_text[:len(sample_text)-1]
    sample_text = sample_text.replace('<br />', '')
    sample_text = sample_text.replace("\\'", "'")
    sample_text = sample_text.replace('\n', '')
    for spaced in [',', '!', '?', '—', ':']:
        sample_text = sample_text.replace(spaced, ' {0} '.format(spaced))
    sample_text = sample_text.replace("(", "( ")
    sample_text = sample_text.replace(")", " ) ")
    sample_text = sample_text.replace("  ", " ")

    sample_text = sample_text.split(' ')
    sample2 = []
    for word in sample_text:
        if word != '':
            sample2.append(word)

    return sample2

########################################################################################
# function update_nw_vec takes two next word dictionaries and combines them
# Example:  If the key 'movie is so' has values {'good': 1, 'bad': 2}
#           and {'good': 2, 'scary': 1} in dict1 and dict2 respectively,
#           then the 'movie is so' value will be updated to {'good': 3, 'bad': 2, 'scary': 1}.

# Input: dict1, dict2 = 2 next word dictionaries

# Output: dict3 = updated dictionary combining next word information from dict1 and dict2
########################################################################################

def update_nw_vec(dict1, dict2):
    dict3 = {words: dict1.get(words, 0) + dict2.get(words, 0) for words in set(dict1).union(dict2)}

    return dict3

########################################################################################
# function update_nwd takes two dictionaries from two different reviews and combines them
#           by either updating an existing key with new next word data or making a new key, value.
#           Only distinct keys are considered, since it is a dictionary.
#           So non-repeating keys are added and repeated keys are updated by update_nw_vec.

# Input: dict1, dict2 = 2 nested next word dictionaries from 2 reviews

# Output: dict3 = updated dictionary combining next word information from dict1 and dict2
########################################################################################


def update_nwd(dict1, dict2):
    dict3 = dict2
    for key in dict1.keys():
        if key in dict2.keys():
            dict3[key] = update_nw_vec(dict1[key], dict2[key])
        if key not in dict2.keys():
            dict3[key] = dict1[key]

    return dict3

########################################################################################
# function next_word: Picks the next word to appear in a sentence based on the next word dictionary data

# Input: nwd = a dictionary containing all k word sequences as keys and the next word probabilities as values
#               as generated by next_word_vectors, normalize_vec, update_nw_vec, and update_nwd

# Input: word_seq = the k word sequence which the function will find a probable next word for

# Ouput: weighted_choice = a randomly chosen next word which has appeared after word_seq in the data set
#                          chosen with weights based on number of times the next word has appeared after word_seq
########################################################################################

def next_word(nwd, word_seq):
    word_seq_choices = nwd[word_seq]
    word_choices = list(word_seq_choices.keys())
    probs = list(word_seq_choices.values())
    weighted_choice = choice(word_choices, 1, True, probs)
    return weighted_choice

########################################################################################
# function sentence_chain: takes a seed sequence of words and generates a movie review with a set number of sentences.
#                           Ends prematurely if the next word is the ending token "<END>".

# Input: next_word_dict = a nwd generated as above
# Input: seed = the seed string which starts the generated review
# Input: sentence_num = an integer, the number of sentences the review should have.  Note: based on number of periods,
#                                   so currently ellipses count as 3 sentences.

# Output: sentence = a string which contains the randomly generated movie review using the next_word function
########################################################################################

def sentence_chain(next_word_dict, seed, sentence_num):
    current_words = seed
    # print(current_words)
    sentence = seed
    i = 0
    while i < sentence_num:
        sentence += ' '
        nextword = next_word(next_word_dict, current_words)
        if nextword[0] == '<END>':
            break
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