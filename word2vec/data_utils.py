import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
import os
import pdb
import re
import pandas as pd
from tqdm import tqdm
from itertools import chain
import numpy as np
import json


def clean_data(parent_folder, store=False, filename=None):
    """
    Given the absolute path to a folder, reads all the txt files from within,
    cleans them [removing punctuations,
                 removing stop words,
                 changing cases to small,
                 removing words containing non-alphabetical charecters]
    and stores each of the cleaned sentences in a list and returns the list.
    input:
        parent_folder: string containing the absolute path
        store : if true, stores the result in file.
        filename : String containing the name in which to store the result

    output:
        cleaned_list : List containing the cleaned sentences.
    """
    assert os.path.isdir(parent_folder), "Folder does not exist."
    assert store and filename is not None or not store, "Filename cannot be empty!"

    # read files

    stop_words = set(stopwords.words('english'))
    pattern = re.compile(r'[^a-zA-Z.\' ]')
    blank_pattern = re.compile(' +')
    line_break_pattern = re.compile(r'[\r\n]')
    char_pattern = re.compile(r'[^a-zA-Z]')
    filelist = None
    for root, dirs, file in os.walk(parent_folder):

        filelist = file
        break

    corpus_list = []

    for f in tqdm(filelist):

        fo = open(os.path.join(parent_folder, f), 'rb')
        data = fo.read()

        data = data.decode('latin-1')

        clean_data1 = re.sub(line_break_pattern, ' ', data)
        clean_data2 = re.sub(blank_pattern, ' ', clean_data1)
        clean_data3 = re.sub(pattern, '', clean_data2)
        clean_data4 = re.sub(blank_pattern, ' ', clean_data3)

        sentences = nltk.sent_tokenize(clean_data3)
        for sentence in sentences:

            # print('Original sentence :', sentence)
            filtered_sentence = []
            word_list = sentence.strip().split(' ')
            # print('The word_list :', word_list)
            for word in word_list:

                if word not in stop_words:

                    filtered_sentence.append(
                        re.sub(char_pattern, '', word.lower()))

            corpus_list.append(filtered_sentence)
            # print('The filtered_sentence :', filtered_sentence)
            # pdb.set_trace()

    if store:
        with open(filename, 'w') as f:
            json.dump(corpus_list, f)

    return corpus_list


def create_vocab(corpus_list, store=False,
                 vocab_list_name=None,
                 vocab_dict_name=None):
    '''
    Given a list of sentences or a filename that contains a list of sentences
    creates and returns an ordered list containing all the words and a dictionary with the
    keys as the words and values as the index of the word in the vocabulary list

    input:
        copus_list : A list of sentences/or a filename containing the same

    output:
        vocab_list : An ordered list containing the words.
        vocab_dict : A dictionary contaning the words and their index in the
                     vocab_list
    '''
    assert isinstance(corpus_list, (list, str)), "corpus_list should either be a \
list containing the list of words or a path to the filename containing the same."

    assert store and vocab_list_name is not None or not store, "Filename cannot be empty!"
    assert store and vocab_dict_name is not None or not store, "Filename cannot be empty!"

    if isinstance(corpus_list, str):
        assert os.path.isfile(corpus_list), "File does not exist."
        fp = open(corpus_list)
        corpus_list = json.load(fp)

    pdb.set_trace()
    # create the vocab list
    vocab_list = list(set(chain.from_iterable(corpus_list)))
    vocab_list.sort()
    # create a dictionary for faster access of each words in the vocabulary
    vocab_dict = {}
    for i in range(len(vocab_list)):
        vocab_dict[vocab_list[i]] = i

    # store the vocab and the vocab dictionary?
    if store:
        with open(vocab_list_name, "w") as fp:
            json.dump(vocab_list, fp)

        with open(vocab_dict_name, "w") as fp:
            json.dump(vocab_dict, fp)

    return vocab_list, vocab_dict


def create_cbow_dataset(corpus_list, save_filename=None, context_window=2):
    '''
    Creates the training and testing data for CBOW based word2vec training
    input:
        corpus_list : A list of sentences (in the form of word lists)/the name
                      of the file containing the corpus list
        context_window : A integer containing the length of the context window

    ouput:
        cbow_dataset : A list where each entry is a list of words of the following format
                        [ inp_word1, inp_word2, inp_word3, . . ., outputword]
                        where the number of words in the input is based on the
                        size of the context window
    '''
    assert isinstance(corpus_list, (list, str)), "corpus_list should either be a \
    list containing the list of words or a path to the filename containing the same."

    assert save_filename is not None, "Filename cannot be empty!"

    if isinstance(corpus_list, str):
        assert os.path.isfile(corpus_list), "File does not exist."
        fp = open(corpus_list)
        corpus_list = json.load(fp)

    cbow_dataset = []

    for sentence in tqdm(corpus_list):

        training_tuples = get_cbow_training_tuples(
            sentence, context_window=context_window)

        for tuple_val in training_tuples:

            cbow_dataset.append(tuple_val)

    # create a json file
    with open(save_filename, "w") as fp:
        json.dump(cbow_dataset, fp)

    return cbow_dataset


def get_cbow_training_tuples(word_list, context_window=2):
    '''
    Creates a set of training tuples from a single sentence.
    input:
        word_list : a list containing words.
        context_window : integer containing the size of the context
                         window

    output:
        training_tuples: a list of lists where each sub list is a
                         list of words in the following form:
                         [ inp_word1, inp_word2, inp_word3, . . ., outputword]
                        where the number of words in the input is based on the
                        size of the context window
    '''
    training_tuples = []
    sentence_length = len(word_list)
    for i in range(sentence_length):

        if sentence_length > 1:
            context_words = []
            index_word = word_list[i]
            context_window_left_limit = max(0, i - context_window)
            context_window_right_limit = min(
                sentence_length - 1, i + context_window)

            # add words from the left side of the context window
            left_indices = np.arange(context_window_left_limit, i)
            for j in range(left_indices.shape[0]):
                context_words.append(word_list[left_indices[j]])

            # add words from the right side of the context window
            right_indices = np.arange(i + 1, context_window_right_limit + 1)
            for j in range(right_indices.shape[0]):
                context_words.append(word_list[right_indices[j]])

            context_words.append(index_word)
            training_tuples.append(context_words)

        # pdb.set_trace()
    return training_tuples


def create_skipgram_dataset(corpus_list, save_filename=None, context_window=2):
    '''
    Creates the training and testing data for Skip Gram based word2vec training
    input:
        corpus_list : A list of sentences (in the form of word lists)/the name
                      of the file containing the corpus list
        context_window : A integer containing the length of the context window

    ouput:
        skipgram_dataset : A list where each entry is a list of words of the following format
                        [ inp_word, outputword]
    '''
    assert isinstance(corpus_list, (list, str)), "corpus_list should either be a\
    list containing the list of words or a path to the filename containing the same."

    assert save_filename is not None, "Filename cannot be empty!"

    if isinstance(corpus_list, str):
        assert os.path.isfile(corpus_list), "File does not exist."
        fp = open(corpus_list)
        corpus_list = json.load(fp)

    skipgram_dataset = set()

    for sentence in tqdm(corpus_list):
        skipgram_dataset.update(get_skipgram_training_tuples(
            sentence, context_window=context_window))

    skipgram_dataset = [list(i) for i in skipgram_dataset]
    # create a json file
    with open(save_filename, "w") as fp:
        json.dump(skipgram_dataset, fp)

    return skipgram_dataset


def get_skipgram_training_tuples(word_list, context_window):
    '''
    Creates a set of training tuples from a single sentence.
    input:
        word_list : a list containing words.
        context_window : integer containing the size of the context
                         window

    output:
        training_tuples: a list of lists where each sub list is a
                         list of words in the following form:
                         [ inp_word, outputword]
    '''

    # Input-Output word list
    output_list = set()
    for idx, word in enumerate(word_list):
        input = word

        # The left side words of the main word
        left_context = word_list[max(0, idx - context_window):idx]

        # The right side words of the main word
        right_context = word_list[(idx + 1):(idx + context_window + 1)]

        # Append both left and right side context words
        output_list.update((input, lc) for lc in left_context)
        output_list.update((input, rc) for rc in right_context)

    return output_list


if __name__ == '__main__':

    #clean_data('./gutenberg', store=True, filename='part_gutenberg.json')
    # create_vocab('./mini_gutenberg.json', store=True, vocab_dict_name='part_gutenberg_dict.json',
    #             vocab_list_name='part_gutenberg_list.json')
    '''
    with open('./part_gutenberg_list', "rb") as fp:
        vocab_list = pickle.load(fp)

    with open('./part_gutenberg_dict', "rb") as fp:
        vocab_dict = pickle.load(fp)

    pdb.set_trace()
    '''
    create_cbow_dataset('./data/part_gutenberg.json',
                        'cbow_style_dataset_part_gutenberg.json')
    '''
    create_skipgram_dataset('./mini_gutenberg.json',
                            'skipgram_style_training_dataset_nr.json')
    '''