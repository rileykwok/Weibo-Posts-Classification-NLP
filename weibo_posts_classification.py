#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:24:57 2018

@author: Andy Wang, Kelvin Kwok, Riley Kwok

"""
# READ DATA

import pandas as pd

df1 = pd.read_json("data//Status_Data1.json", encoding= 'UTF-8')
df2 = pd.read_json("data//Status_Data2.json", encoding= 'UTF-8')
df3 = pd.read_json("data//Status_Data3.json", encoding= 'UTF-8')
df4 = pd.read_json("data//Status_Data4.json", encoding= 'UTF-8')
df5 = pd.read_json("data//Status_Data5.json", encoding= 'UTF-8')
df6 = pd.read_json("data//Status_Data6.json", encoding= 'UTF-8')
df7 = pd.read_json("data//Status_Data7.json", encoding= 'UTF-8')
df8 = pd.read_json("data//Status_Data8.json", encoding= 'UTF-8')
df9_comcity = pd.read_json("data//Status_Data9_Comment_Data_with_city_code.json", encoding= 'UTF-8')
# df9_city = pd.read_json("Status_Data9_with_city_code.json", error_bad_lines = "False")
province = pd.read_json("data//weibo-china-province-city.json")
task_status = pd.read_json("data//task-status.json", encoding= 'UTF-8')
task_status['text'] = task_status['s_text']
task_status['user_id'] = task_status['uid']
task_status['id'] = task_status['sid']
task_status = task_status.drop(columns = ['s_text', 'uid', 'sid'])

label = pd.read_csv("data//train_label.txt", delimiter = '	', header = None, names = ['ID', 'other', 'category'])

frames = [df1, df2, df3, df4, df5, df6, df7, df8, task_status]
df = pd.concat(frames)
df = df.reset_index(inplace = False).drop(columns = ['index'])

category_dict = dict(zip(label.ID, label.category))

df['label'] = df.id.map(category_dict)

# DATA CLEANING

import re
import jieba

def rm_char(text): #remove symbols and special characters
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text

def get_stop_words(): # load stopwords from txt file
    file = open('stopwords-zh.txt','rb').read().decode('utf8').split('\n')
    return set(file)

def rm_tokens(words): # remove stop words and numbers
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list

def convert_doc_to_wordlist(doc, cut_all = False):
    s_list = re.split('，|【|】|。', doc)
    s_list = map(rm_char, s_list)
    s_list_cut = [rm_tokens(jieba.cut(s, cut_all = cut_all)) for s in s_list]
    word_list = sum(s_list_cut, [])
    return word_list

l = []
x = []
for i in range(len(df)): #segment sentences to words
    l.append(convert_doc_to_wordlist(df.text.iloc[i]))
for j in l: # join cleaned words back to sentence
    x.append(' '.join(j))

df['new'] = pd.DataFrame(x)

df_train = df[df.label.notnull()]
df_test = df[df.label.isnull()]

########### Supervised Modelling ############

# load pre-trained word embeddings wording
with open('sgns.weibo.word', 'rb') as f:
    for line_b in f:
        line_u = line_b.decode('utf-8')
        
embedding = pd.read_csv('sgns.weibo.word', header = None, sep = ' ')
embedding = embedding.drop(301, 1)
embedding = embedding.set_index(0)

# create a weight matrix for words in training docs
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
from keras.utils import to_categorical

t = Tokenizer()
t.fit_on_texts(df.new)
vocab_size = len(t.word_index)+1
encoded_docs = t.texts_to_sequences(df_train.new)

labels_keras = np.array(df_train.label)
y_label = to_categorical(labels_keras, num_classes = 13)

max_length = 200
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

embeddings_index = dict()
for i in range(len(embedding)):
	word = embedding.index[i]
	coefs = embedding.iloc[i,:].astype('float32')
	embeddings_index[word] = coefs


embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length= max_length, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(13, activation='softmax'))

model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=[f1])

history = model.fit(padded_docs, y_label, epochs=20, verbose=1, batch_size = 16, validation_split=0.3)

import matplotlib.pyplot as plt
def modelplot(history):
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.axis([0,20,0.3,1])
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.axis([0,20,0,4])
    plt.show()

modelplot(history)

'''
encoded_docs = t.texts_to_sequences(df_test.new)
padded_docs_y = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
model.predict()
'''

#####################################
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = None #can be any integer i.e. 20, 50 etc. If None, use all words 

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features= no_features)
tfidf_vectorizer.fit(df.new)
tfidf = tfidf_vectorizer.transform(df_train.new)
tfidf_real = tfidf_vectorizer.transform(df_test.new)

tfidf_feature_names = tfidf_vectorizer.get_feature_names() 
tfidf_vectorizer.idf_ 
tfidf.toarray()  

# LDA
tf_vectorizer = CountVectorizer(max_features=no_features)
tf_vectorizer.fit(df.new)
tf = tf_vectorizer.transform(df_train.new)
tf_feature_names = tf_vectorizer.get_feature_names() 
tf.toarray()

# Neural network
model = Sequential()
e = Embedding(1438, 10173, weights=[dense], trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(10173, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(13, activation='softmax'))

model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(tfidf, y_label, epochs=20, verbose=1,validation_split=0.3)



# Run NMF
from sklearn.decomposition import NMF, LatentDirichletAllocation
no_topics = 13
nmf = NMF(n_components=no_topics, init='nndsvd').fit(tfidf)
W = nmf.fit_transform(tfidf)
H = nmf.components_

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, learning_method='online', learning_offset=50.).fit(tf)
W_lda = lda.fit_transform(tf)
H_lda = lda.components_


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 15
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)


#######################################
tfidf_df = pd.DataFrame(tfidf.toarray())
dummies = pd.get_dummies(df_train.user_id)
dummies = dummies.reset_index(inplace = False).drop(columns = ['index'])
tfidf_new = pd.concat([tfidf_df, dummies], axis = 1)



from sklearn.model_selection import train_test_split
y = df_train.label
x_train, x_test, y_train, y_test = train_test_split(tfidf_new, y, test_size=0.3, random_state=1)


import xgboost as xgb
m = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1)
train1 = m.fit(x_train, y_train)
predict = m.predict(x_test)

m1 = xgb.XGBClassifier()
train2 = m1.fit(x_train, y_train)
predict2 = m1.predict(x_test)


from sklearn.metrics import f1_score
f1_score(y_test, predict2, average = 'micro')


from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, predict2)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, predict2)


from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from catboost import CatBoost

m_cat = CatBoostClassifier(iterations = 2000, loss_function = 'MultiClass') 
m_cat.fit(x_train, y_train) 
y_pred_cat = m_cat.predict(x_test)


from sklearn.metrics import f1_score
f1_score(y_test, y_pred_cat, average = 'micro')


tfidf_df = pd.DataFrame(tfidf_real.toarray())
dummies = pd.get_dummies(df_test.user_id)
dummies = dummies.reset_index(inplace = False).drop(columns = ['index'])
x_true = pd.concat([tfidf_df, dummies], axis = 1)

y_pred_actual = m_cat.predict(x_true)
y_pred_actual = pd.DataFrame(y_pred_actual)
df_test_pred = pd.concat([df_test.reset_index(inplace = False).drop(columns = ['index']), y_pred_actual], axis = 1)

df_test_pred['label'] = df_test_pred.iloc[:,-1]
df_test_pred = df_test_pred.drop([0], axis=1) 

########
from sklearn.ensemble import RandomForestClassifier
m_rand = RandomForestClassifier()
m_rand.fit(x_train, y_train)
predict_rand = m_rand.predict(x_test)

from sklearn.metrics import f1_score
f1_score(y_test, predict_rand, average = 'micro')

y_pred_actual_rand = m_rand.predict(x_true)

############################################

final_df = pd.concat([df_test_pred, df_train])
final_df = final_df.reset_index(inplace = False).drop(columns = ['index'])

# Saving files

final_df.to_csv("Final_2.csv",encoding='utf-8-sig')
final1.to_csv('team_2.txt', sep='\t', index=False)