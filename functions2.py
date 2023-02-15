from numpy.random import choice

def dict_text(text):

    distinct_stw = list(set(text))
    word_indx_dict = {i: word for i, word in enumerate(distinct_stw)}

    return distinct_stw, word_indx_dict


def k_set_word_list(text, k):

    sets_of_k_words = [' '.join(text[i:i+k])
                       for i, _ in enumerate(text[:-k+1])]
    distinct_k_word_sets = list(set(sets_of_k_words))

    return sets_of_k_words, distinct_k_word_sets


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

def normalize_vec(dict):
    for word in dict.keys():
        sum = 0
        for next_word in dict[word].keys():
            sum += (dict[word])[next_word]
        for next_word in dict[word].keys():
            (dict[word])[next_word] = float((dict[word])[next_word])/float(sum)

    return dict

def new_text_pre(sample_text):
    sample_text += ' <END>'
    if sample_text[:2] == "b'" or sample_text[:2] == 'b"':
        sample_text = sample_text[2:]
    if sample_text[-1] == "'" or sample_text[-1] == '"':
        sample_text = sample_text[:len(sample_text)-1]
    sample_text = sample_text.replace('<br />', '')
    sample_text = sample_text.replace("\\'", "'")
    sample_text = sample_text.replace('\n', '')
    for spaced in ['.', ',', '!', '?', '—', ':']:
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


def update_nw_vec(dict1, dict2):
    dict3 = {words: dict1.get(words, 0) + dict2.get(words,0) for words in set(dict1).union(dict2)}

    return dict3

def update_nwd(dict1, dict2):
    dict3 = dict2
    for key in dict1.keys():
        if key in dict2.keys():
            dict3[key] = update_nw_vec(dict1[key], dict2[key])
        if key not in dict2.keys():
            dict3[key] = dict1[key]

    return dict3

def next_word(nwd, word_seq):
    word_seq_choices = nwd[word_seq]
    word_choices = list(word_seq_choices.keys())
    probs = list(word_seq_choices.values())
    weighted_choice = choice(word_choices, 1, True, probs)
    return weighted_choice

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