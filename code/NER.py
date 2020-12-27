import numpy as np
import pickle
import nltk
from nltk.tag.stanford import StanfordNERTagger

def NER_tagger(text, model_path, jar_path):

    # Prepare NER tagger with english model
    ner_tagger = StanfordNERTagger(model_path, jar_path, encoding='utf-8')

    # Run NER tagger on words
    return ner_tagger.tag(text)


########################################################################
#   This function takes the text as input and outputs a python list
#       containing the dictionary of each sentence with words as
#       keys and the POS tags as values.

#       sentences: contains a dictionary of the vocabs as keys and their
#       POS tags list as their values.
########################################################################
def text_tokenize(path):
    print("**************************************************************\n"
          "Preparing the text\n"
          "**************************************************************")
    file = open(path, 'r')
    lines = file.readlines()
    X_test, Y_test = [], []
    temp_x, temp_y = [], []

    for line in lines:
        if len(line) == 2:
            X_test.append(temp_x)
            Y_test.append(temp_y)
            temp_x, temp_y = [], []

        else:
            words = line.split()
            temp_y.append(words[1])
            temp_x.append(words[0])

    print("**************************************************************\n"
          "Text has been prepared\n"
          "**************************************************************")
    return X_test, Y_test


def text_tokenize1(path):
    print("**************************************************************\n"
          "Preparing the text\n"
          "**************************************************************")
    file = open(path, 'r')
    lines = file.readlines()
    X_test, Y_test = [], []

    for line in lines:
        if len(line) > 2:
            words = line.split()
            X_test.append(words[0])
            Y_test.append(words[1])

    print("**************************************************************\n"
          "Text has been prepared\n"
          "**************************************************************")
    return X_test, Y_test