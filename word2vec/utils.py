import numpy as np
import os


def get_one_hot_vec(word, vocab_dict):
    '''
    Returns one hot vector for the given word
    vocab_dict - List of all words
    word - the input word
    '''
    ohv = np.zeros(len(vocab_dict))
    idx = vocab_dict[word]
    ohv[idx] = 1
    return ohv
