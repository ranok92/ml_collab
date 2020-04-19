import torch
import numpy as np 
import pdb


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




def test_embedding_question_words(w2vec_embeddings, word_file_name):
    '''
    Performs a cosine similarity between the vector obtained by vector algebra operations
    done on the first 3 words of a tuple and the vector representation of the 4th word.

    input:
        w2vec_embeddings : a dictionary containing the word embeddings?
                           key - value : word - word-embedding

        question_word_list : a list where each value is in turn is a list of 4 words 
                             from the tensorflow question-word.txt file.
        
    output :
        total_cosine_similarity : float containing the sum of the cosine similarity 
                                  obtained from all the tuples in the list
    '''
    #read the file 
    fp = open(word_list_filename, 'r')
    question_word_list = []
    word_list
    for line in fp:

        if len(line.split(' '))!=4:
            continue
        else:
            2d_word_list.append(line.split(' '))
    

    for test_tuple in question_word_list:
        return 0


if __name__=='__main__':

    read_question_wordlist('./questions-words.txt')