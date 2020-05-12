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
from collections import OrderedDict


def clean_data(parent_folder, store=False, filename=None, freqency_threshold=1):
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

    stop_words = set(stopwords.words('english'))
    pattern = re.compile(r'[^a-zA-Z.\' ]')
    blank_pattern = re.compile(' +')
    line_break_pattern = re.compile(r'[\r\n]')
    char_pattern = re.compile(r'[^a-zA-Z]')
    filelist = None

    # maintain a a frequency dictionary
    freq_dict = {}

    # read files
    for root, dirs, file in os.walk(parent_folder):
        filelist = file
        break

    corpus_list = []

    print("Generating Sentences")
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

            #print('Original sentence :', sentence)
            filtered_sentence = []
            word_list = sentence.strip().split(' ')

            # print('The word_list :', word_list)
            for word in word_list:
                if word not in stop_words:
                    filtered_sentence.append(
                        re.sub(char_pattern, '', word.lower()))

            for filt_words in filtered_sentence:
                if filt_words in freq_dict.keys():
                    freq_dict[filt_words] += 1
                else:
                    freq_dict[filt_words] = 1

            corpus_list.append(filtered_sentence)

    sorted_freq_dict = OrderedDict(sorted(freq_dict.items(), key=lambda t: t[1]))
    low_freq_words = []
    corpus_list_high_freq = []

    for sentence in corpus_list:
        contains_lowFreq = False
        for word in sentence:
            if sorted_freq_dict[word] < freqency_threshold:
                contains_lowFreq = True
                break

        if not contains_lowFreq:
            corpus_list_high_freq.append(sentence)

    print("Original corpus list length :{}".format(len(corpus_list)))
    print("Length of corpus of high frequency words :{}".format(len(corpus_list_high_freq)))

    if store:
        with open(filename, 'w') as f:
            json.dump(corpus_list_high_freq, f)

    return corpus_list, corpus_list_high_freq, sorted_freq_dict


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

    # get the word frequencies
    print("Getting word frequencies")
    for sentence in tqdm(corpus_list):
        for word in sentence:
            if word not in frequency_dict.keys():
                frequency_dict[word] = 1
            else:
                frequency_dict[word] += 1

    print("Building dataset")
    for sentence in tqdm(corpus_list):
        training_tuples = get_cbow_training_tuples(sentence, context_window=context_window)
        for tuple_val in training_tuples:
            cbow_dataset.append(tuple_val)

    # create a json file
    with open(save_filename, "w") as fp:
        json.dump({'dataset': cbow_dataset, 'freq_dict': frequency_dict}, fp)

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

    return training_tuples


def create_skipgram_dataset(corpus_list, save_filename=None, context_window=2):
    '''
    Creates the training and testing data for Skip Gram based word2vec training
    input:
        corpus_list : A list of sentences (in the form of word lists)/the name
                      of the file containing the corpus list
        context_window : A integer containing the length of the context window

    ouput:
        dataset : A dictionary containing two things:
            {'dataset' : skipgram_dataset, 'freq_dict' : freq_dict}
            skipgram_dataset : A list where each entry is a list of words of the following format
                            [ inp_word, outputword]
            freq_dict : A dictionary containing the frequency of occurence of each of the words
                        { word : frequency }
    '''
    assert isinstance(corpus_list, (list, str)), "corpus_list should either be a\
    list containing the list of words or a path to the filename containing the same."

    assert save_filename is not None, "Filename cannot be empty!"

    if isinstance(corpus_list, str):
        assert os.path.isfile(corpus_list), "File does not exist."
        fp = open(corpus_list)
        corpus_list = json.load(fp)

    skipgram_dataset = set()
    frequency_dict = {}

    # get the word frequencies
    print("Getting word frequencies")
    for sentence in tqdm(corpus_list):
        for word in sentence:
            if word not in frequency_dict.keys():
                frequency_dict[word] = 1
            else:
                frequency_dict[word] += 1

    # create the dataset
    print("Building dataset")
    for sentence in tqdm(corpus_list):
        skipgram_dataset.update(get_skipgram_training_tuples(
            sentence, context_window=context_window))

    skipgram_dataset = [list(i) for i in skipgram_dataset]

    # create a JSON file
    with open(save_filename, "w") as fp:
        json.dump({'dataset': skipgram_dataset, 'freq_dict': frequency_dict}, fp)

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

    parser = argparse.ArgumentParser()

    # parent folder contains all the txt files
    parser.add_argument("--parent_folder", type=str, required=True, help="path to the parent folder")
    parser.add_argument("--data_store_path", type=str, required=True, help="path to store the dataset")
    parser.add_argument("--freq_threshold", type=int, default=100, help="frequecy threshold cutoff")
    parser.add_argument("--context_window", type=int, default=2, help="context window size")
    opt = parser.parse_args()

    clean_data(opt.parent_folder, store=True,
                filename=os.path.join(opt.data_store_path, 'clean_data.json'),
                freqency_threshold=opt.freq_threshold)

    create_cbow_dataset(os.path.join(opt.data_store_path, 'clean_data.json'),
                        save_filename=os.path.join(opt.data_store_path, 'cbow_style_training_dataset.json'),
                        context_window=opt.context_window)

    create_skipgram_dataset(os.path.join(opt.data_store_path, 'clean_data.json'),
                            save_filename=os.path.join(opt.data_store_path,'skipgram_style_training_dataset.json'),
                            context_window=opt.context_window)
