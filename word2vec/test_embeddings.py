import torch
import numpy as np 
import ipdb
import os
import json
from scipy import spatial

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
    
    for test_tuple in question_list:
        #perform the testing operation here

        #an array that stores the embeddings of the words in the current test_tuple
        vector_array = np.zeros((4, embedding_array.shape[1])) 
        for i in range(len(test_tuple)):
            
            try:
                vector_array[i, :] = embedding_array[vocab_dict[test_tuple[i]]]
            except KeyError:
                print ("Could not find: '{}'. Skipping the entire tuple.".format(test_tuple[i]))
                break

        ipdb.set_trace()
        cosine_similarity = get_cosine_similarity(vector_array)


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
    return spatial.distance.cosine((vector_array[0, :] - vector_array[1, :] + \
                                    vector_array[2, :]), vector_array[3, :]) 
   



if __name__=='__main__':

    test_embedding_question_words('./data/embeddings.json', 
                                  './data/questions-words.txt')