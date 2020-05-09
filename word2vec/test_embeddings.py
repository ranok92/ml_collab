import torch
import numpy as np
import ipdb
import os
import json
from scipy import spatial
from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.models import KeyedVectors



def get_word_embedding_dictionary(vocab_dictionary, word_em_wt_matrix):
    '''
    Creates a w2vec_embedding dictionary of the following format:
        key - value : word - word_embedding

    input:
        vocab_dictionary : A dict{} key - value : word - word_index (in the vocab)
        word_em_wt_matrix : The weight matrix/first layer of the network trained
                            to get the embeddings.


    output:
        word_em_dict : A dict key - value : word - word_embedding
    '''
    return 0


def read_question_wordlist(word_list_filename):
    '''
    Given the filename to the text file containing the question_words,
    [http://download.tensorflow.org/data/questions-words.txt]
    Creates a 2d list of the same.
    input :
        word_list_filename : String containing the path to the file containing
                             the question words.

    output :
        question_word_list : A 2d list containing the question words.
    '''

    fp = open(word_list_filename, 'r')

    word_list
    for line in fp:

        if len(line.split(' '))!=4:
            continue
        else:
            print(line)
        pdb.set_trace()




def test_embedding_question_words(w2vec_embeddings_json, question_word_list):
    '''
    Performs a cosine similarity between the vector obtained by vector algebra operations
    done on the first 3 words of a tuple and the vector representation of the 4th word.

    input:
        w2vec_embeddings_json : a json file containing the word embeddings and the
                            vocab_dictionary.
                            Can be a filename or the dictionary itself
                            keys : <'vocab_dict', 'embedding'>
                            vocab_dict : contains a dictionary for the words in the
                                         vocabulary and their index values

                            embedding : a 2d list of size no_of_words x embedding_size


        question_word_list : a list where each value is in turn is a list of 4 words
                             from the tensorflow question-word.txt file.

    output :
        total_cosine_similarity : float containing the sum of the cosine similarity
                                  obtained from all the tuples in the list
    '''
    #read the word embedding file

    if type(w2vec_embeddings_json)==str:
        assert os.path.isfile(w2vec_embeddings_json), "File does not exist."
        fp_embedding = open(w2vec_embeddings_json)
        w2vec_embeddings_json = json.load(fp_embedding)

    embedding_list = w2vec_embeddings_json['embedding']
    vocab_dict = w2vec_embeddings_json['vocab_dict']

    embedding_array = np.asarray(embedding_list)

    #read the question_word_list file

    assert os.path.isfile(question_word_list), "File does not exist."

    fp_qwords= open(question_word_list, 'r')
    question_list = []
    cosine_similarity = 0
    tuples_skipped = 0
    #select the lines that are of the 4 word format
    for line in fp_qwords:

        if len(line.split(' '))!=4:
            continue
        else:
            line_list =  line.split(' ')
            stripped_words = []
            for word in line_list:
                stripped_words.append(word.strip())

            question_list.append(stripped_words)
    total_tuples = len(question_list)

    for test_tuple in question_list:
        #perform the testing operation here

        #an array that stores the embeddings of the words in the current test_tuple
        vector_array = np.zeros((4, embedding_array.shape[1]))
        for i in range(len(test_tuple)):

            skip = False
            try:
                vector_array[i, :] = embedding_array[vocab_dict[test_tuple[i]]]
            except KeyError:
                #print ("Could not find: '{}'. Skipping the entire tuple.".format(test_tuple[i]))
                tuples_skipped += 1
                skip = True
                break

        if not skip:
            cosine_similarity += get_cosine_similarity(vector_array)
    print('Out of the initial {} tuples, {} tuples had atleast one word which was \
not present in the vocabulary.'.format(total_tuples, tuples_skipped))
    return cosine_similarity/(total_tuples-tuples_skipped)


def get_cosine_similarity(vector_array):
    '''
    Given an array containing the embedding vector of the words, returns the
    cosine similarity between the vector generated from vector algebra and the
    vector obtained from the embedding:

    input:
        vector_array : A 2 dim array of size (4 x embedding vector length)

    output
        cosine_similarity : float containing the cosine similarity calculated using the
        following formula
        cos ((word1-word2+word3), word4)

    '''
    return 1 - spatial.distance.cosine((vector_array[1, :] - vector_array[0, :] + \
                                    vector_array[2, :]), vector_array[3, :])


def get_k_most_similar(word, w2vec_embeddings_json, k=10):
    '''
    Given a word, find the k most similar words in vocabulary using.
    We use cosine similarity to get the similarity between words.

    input:
        word: The input word

    output:
        A list of tuples containing  k most similar words and their 
            similarity obtained from vocabulary

    '''
    if type(w2vec_embeddings_json)==str:
        assert os.path.isfile(w2vec_embeddings_json), "File does not exist."
        fp_embedding = open(w2vec_embeddings_json)
        w2vec_embeddings_json = json.load(fp_embedding)

    embedding_list = w2vec_embeddings_json['embedding'] # Get the embedding
    embedding_array = np.asarray(embedding_list) # Convert it to numpy array
    vocab_dict = w2vec_embeddings_json['vocab_dict'] # Vocab dictionary

    try:
        word_vector = embedding_array[vocab_dict[word]] # Get the word vector
        distance_dict = {} # To store all words similarity
        for key in vocab_dict: # Loop for all words in vocabulary
            if key != word: # Check if word is not the same
                key_vector = embedding_array[vocab_dict[key]]
                cos_sim = 1 - spatial.distance.cosine(word_vector, key_vector)
                distance_dict[key] = cos_sim
        # Sort the
        k_similar_words = sorted(distance_dict, key=distance_dict.get, reverse=True)[:k]
        similar_dict = []

        for word in k_similar_words:
            similar_dict.append((word, distance_dict[word]))

        return similar_dict
    except KeyError:
        print("Word not found in vocabulary.")




def compare_with_word2vec(word_list, w2vec_embeddings_json, 
                          reference_filename,
                          k=10):
    '''
    Given the wordlist returns the n most similar words.
        input:
            word_list ; list of words
            w2vec_embeddings_json : filename of json obtained from training
            reference_filename : filename of word2vec reference
            k : number of most similar words to be extracted
    '''
    #load the Google word2vec
    assert os.path.isfile(reference_filename), "File does not exist."

    if type(w2vec_embeddings_json)==str:
        assert os.path.isfile(w2vec_embeddings_json), "File does not exist."
        fp_embedding = open(w2vec_embeddings_json)
        w2vec_embeddings_json = json.load(fp_embedding)

    print("Loading trained model.")
    ref_model_w2v = KeyedVectors.load_word2vec_format(reference_filename , binary=True)
    print("Loading reference model.")

    reference_model = ref_model_w2v.wv
    embedding_list = w2vec_embeddings_json['embedding'] # Get the embedding
    embedding_array = np.asarray(embedding_list) # Convert it to numpy array
    vocab_dict = w2vec_embeddings_json['vocab_dict'] # Vocab dictionary

    #get the words
    master_ref_list = []
    master_sim_list = []
    for word in word_list:

        master_ref_list.append(ref_model_w2v.most_similar(positive=word, topn=k))

        master_sim_list.append(get_k_most_similar(word, w2vec_embeddings_json, k=k))


    return master_ref_list, master_sim_list


if __name__=='__main__':

    '''
    val = test_embedding_question_words('./data/embeddings.json',
                                  './data/questions-words.txt')
    '''
    '''
    reference_filename = '../../NLP/GoogleNews-vectors-negative300.bin'
    w2vec_embeddings_json = './data/embeddings_sgns.json'
    word_list = ['man', 'fruit', 'bicycle', 'school']
    ref, actia = compare_with_word2vec(word_list, w2vec_embeddings_json, 
                          reference_filename,
                          k=10)
    '''
    w2vec_embeddings_json = './data/embeddings_sgns.json'

    most_similar = get_k_most_similar("apple", w2vec_embeddings_json, k=10)
    ipdb.set_trace()
