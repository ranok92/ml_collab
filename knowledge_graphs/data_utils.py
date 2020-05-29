import spacy
import os

NLP = spacy.load('en_core_web_md')

def read_data_from_OMCS(data_file_name):
    '''
    Read data from the omcs dataset and returns a list 
    containing just the sentences in the form of word lists
    input:


    output:

    '''


    return 0


def entity_extraction(list_of_sentences):
    '''
    Reads a list of sentences [[word_list1], [word_list2], [word_list3]...]
    and converts it into subject object

    input:

    output:


    '''
    for sentence in list_of_sentences:
        doc = NLP(sentence)