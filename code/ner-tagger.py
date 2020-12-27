# coding: utf-8

from nltk.tag.stanford import StanfordNERTagger
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
'''
#####################################READ_DATASET#####################################
'''
def data_set(path_file):
    with open(path_file, 'r') as fd:
        x = []
        y = []
        class_dictionary = dict()
        cls_index = 0
        lines = fd.readlines()
        for i in range(len(lines)):
            line = lines[i]
            words = line.split('\t')
            x.append(words[0])
            y.append(words[-1])
        for i in range(len(y)):
            y[i]=y[i].replace('\n','')
            if y[i] == '':
                continue
            else:
                if (class_dictionary.get(y[i]))==None:
                    class_dictionary[y[i]]=cls_index
                    cls_index+=1
        return x,y,class_dictionary
word_test,tag_test,tagDict_test= data_set('Dataset/NERte.txt')
'''
#####################################PREPROSSES#####################################
'''
def preprosses(word,tag):
    train_voc = []
    train_tag = []
    i = 0
    for j in range(i,len(word)):
        if word[j] == '':
            train_tag.append(tag[i:j])
            train_voc.append(word[i:j])
            i = j + 1
    return train_voc,train_tag

X_test,Y_test = preprosses(word_test,tag_test)
classesd = np.unique(tag_test)
index = [0,1]
classess = np.delete(classesd,index)
sorted_labels = sorted(classess, key=lambda name: (name[1:], name[0]))
true_label = sum(Y_test, [])
'''
#####################################CREATE_MODEL#####################################
'''
# jar = 'ner_model/stanford-ner.jar'
# model = 'ner_model/20qk_my-ner-model-eng.ser.gz'
# ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
# '''
# #####################################PREDICT_LABEL#####################################
# '''
# predict_label = []
# for i in range(len(X_test)):
#     labels = ner_tagger.tag(X_test[i])
#     predict_label.append([a[1].strip('[]') for a in labels])
# with open('predict_label.pkl', 'wb') as f:
#      pickle.dump(predict_label, f)

'''
########################################REPORT########################################
'''
with open('predict_label.pkl', 'rb') as f:
     predict_label = pickle.load(f)
per_label = sum(predict_label, [])
# print (metrics.flat_classification_report(Y_test, predict_label, labels=sorted_labels, digits=3))
'''
########################################CONFUSION_MATRIX########################################
'''
cm = confusion_matrix(true_label, per_label, labels=sorted_labels)
np.set_printoptions(suppress=True)
cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
df = pd.DataFrame(cm, columns=sorted_labels,index=sorted_labels)
df.to_csv('conf_matrix.csv')
# print(df)
tp = 0
tn = 0
for i in range(len(true_label)):
    if true_label[i] == per_label[i]:
        tp += 1
    else:
        tn += 1
print(tp,tn)
