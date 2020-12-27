import numpy as np
import re
import pickle

class Viterbi:
    def __init__(self, path):
        self.num_sentences = None
        self.vocabulary = None
        self.vocabs_index = None
        self.num_vocab = None
        self.tags_vocab = None
        self.tags_index = None
        self.num_tags = None
        self.tag_tag_matrix = None
        self.word_tag_matrix = None
        sent, _ = Viterbi.text_tokenize(path)
        self.build_vocab(sent)
        self.fit(sent)

    ########################################################################
    #   This function takes the text as input and outputs a python list
    #       containing the dictionary of each sentence with words as
    #       keys and the POS tags as values.
    #       sentences: contains a dictionary of the vocabs as keys and their
    #       POS tags list as their values.
    ########################################################################
    @staticmethod
    def text_tokenize(path):
        texts = []
        text = ""
        file = open(path, 'r')
        file.readline()
        line = file.readline()
        sentences = []
        sentence = []
        while line:
            words = line.split()
            word = "".join(words[0:-1])
            word.replace(u"\u200C", "")
            word.replace(u"\u0651", "")
            text += word + " "
            sentence.append(tuple((word, words[-1])))
            if line.__contains__(".") or line.__contains__("#"):
                sentences.append(sentence)
                sentence = []
                texts.append(text)
                text = ""
            if words[-1] == '0':
                print(line)
            line = file.readline()
        file.close()
        return sentences, texts

    ########################################################################
    #   This function takes the list of dictionaries and makes a vocabulary
    #       of words.
    #       tags_vocab: ndarray of the the possible tags for the words
    #       vocabulary: ndarray of the possible vocaularies
    ########################################################################
    def build_vocab(self, sentences):
        self.vocabulary = set()
        self.tags_vocab = set()

        num_sentences = len(sentences)
        #   Iterating over all the sentences to make a vocabulary of the tags and the words
        for i, tup in enumerate(sentences):
            if i % 1000 == 0:
                print("building the vocab (%d)" % (int(i / num_sentences * 100)))
            for key, val in tup:
                self.vocabulary.add(key)
                self.tags_vocab.add(val)

        #   Beginning and the end of the sentence should be handled
        self.tags_vocab.add("START")
        self.tags_vocab.add("END")
        self.tags_vocab = [tag for tag in self.tags_vocab]
        self.vocabulary = [word for word in self.vocabulary]
        self.vocabs_index = {word:i for (i, word) in enumerate(self.vocabulary)}
        self.vocabs_index.update({i:word for (i, word) in enumerate(self.vocabulary)})
        self.tags_index = {tag:i for (i, tag) in enumerate(self.tags_vocab)}
        self.tags_index.update({i:tag for (i, tag) in enumerate(self.tags_vocab)})

        self.num_vocab = len(self.vocabulary)
        self.num_tags = len(self.tags_vocab)
        print("**************************************************************\n"
              "The vocabulary has been built\n"
              "**************************************************************")
        return self.vocabulary, self.tags_vocab

    ########################################################################
    #   In this function we learn the parameters of the model including
    #       a conditional tag_tag matrix and a word_tag matrix according
    #       to the provided dataset.
    ########################################################################
    def fit(self, sentences):
        self.tag_tag_matrix = np.zeros((self.num_tags, self.num_tags))
        self.word_tag_matrix = np.zeros((self.num_vocab, self.num_tags))
        num_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            if i % 1000 == 0:
                print("learning the parameters (%d)"% (int(i / num_sentences * 100)))
            for i, (word, tag) in enumerate(sentence):
                self.word_tag_matrix[self.vocabs_index[word], self.tags_index[tag]] += 1
                if i == 0:
                    self.tag_tag_matrix[self.tags_index["START"], self.tags_index[tag]] += 1
                elif i == len(sentence) - 1:
                    self.tag_tag_matrix[self.tags_index[tag], self.tags_index["END"]] += 1
                else:
                    self.tag_tag_matrix[self.tags_index[tag], self.tags_index[sentence[i + 1][1]]] += 1

        # Because of the END tag that is zero in all indexes (No tag after end) we have to set it with 1 to
        #       avoid zero division
        sum = np.sum(self.tag_tag_matrix, axis=1)
        sum[np.where(sum == 0)] = 1
        self.tag_tag_matrix /= np.reshape(sum, (-1, 1))
        self.word_tag_matrix /= np.reshape(np.sum(self.word_tag_matrix, axis=1), (-1, 1))

        print("**************************************************************\n"
              "Model has been learnt\n"
              "**************************************************************")

    ########################################################################
    #   This function takes a sentence as input and tags it
    ########################################################################
    def viterbi(self, sentence):
        # Todo: in this part we take a sentence and POS tag it
        tokens = re.findall(r"[\w']+|[\[=:(%.…'»؛”#؟+?!“×ّ,/)\-،\"_«*\]]", sentence)
        len_sent = len(tokens) + 1
        matrix = np.zeros((self.num_tags, len_sent, 2))
        for i in range(len_sent):
            for j in range(self.num_tags):
                if i == 0:
                    if tokens[0] in self.vocabulary:
                        matrix[j, 0, 0] = 1 * self.tag_tag_matrix[self.tags_index["START"], j] * \
                                      self.word_tag_matrix[self.vocabs_index[tokens[0]], j]
                    else:
                        matrix[j, 0, 0] = 1 * self.tag_tag_matrix[self.tags_index["START"], j] * 1

                elif i == len_sent - 1:
                    temp_array = []
                    for k in range(self.num_tags):
                        temp_array.append(matrix[k, i-1, 0] * self.tag_tag_matrix[k, self.tags_index["END"]])
                    matrix[j, i, 0] = np.max(temp_array)
                    matrix[j, i, 1] = np.argmax(temp_array)
                    break

                else:
                    temp_array = []
                    for k in range(self.num_tags):
                        temp_array.append(matrix[k, i-1, 0] * self.tag_tag_matrix[k, j])

                    if tokens[i] in self.vocabulary:
                        matrix[j, i, 0] = np.max(temp_array) * self.word_tag_matrix[self.vocabs_index[tokens[i]], j]
                    else:
                        matrix[j, i, 0] = np.max(temp_array) * 1
                    matrix[j, i, 1] = np.argmax(temp_array)

        result = []
        for i in range(1, len_sent):
            if i == 1:
                best = int(matrix[0, len_sent - 1, 1])
            else:
                best = int(matrix[best, len_sent - i, 1])
            result.insert(0, (tokens[len_sent - i -1], self.tags_index[best]))
        return result

    ########################################################################
    #   This function calculated the tags for many sentences
    ########################################################################
    def test(self, texts):
        results = []
        num_sent = len(texts)
        for i, sent in enumerate(texts):
            if i % 100 == 0:
                print("Applying the POS tags (%d)"% (int(i / num_sent * 100)))
            results.append(self.viterbi(sent))
        print("**************************************************************\n"
              "Test models has been tagged \n"
              "**************************************************************")
        return results


    def POS_tag_file(self, path):
        file = open(path, "r", encoding="utf-8")
        texts = []
        text = file.readline()
        while text:
            texts.append(text)
            text = file.readline()
        results = self.test(texts)
        return results



