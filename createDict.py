import pandas as pd
import os
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta,SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import gensim
from gensim.models import word2vec

path='./data/qa_test.txt'#数据的路径
path_word2vec='F:\\BaiduNetdiskDownload\\faceData\\word2vec_from_weixin\\word2vec\\word2vec_wx'#word2vec路径
#造数据
fake_data=open(path,'r',encoding='UTF-8').readlines()
tain_data_l=[]
tain_data_r=[]
for line in fake_data:
    for line2 in fake_data:
        if(line is not line2):
            print(line.replace('\n',''),line2.replace('\n',''))
            tain_data_l.append(line.replace('\n',''))
            tain_data_r.append(line2.replace('\n',''))
print('left length:',len(tain_data_l))
print('right length:',len(tain_data_r))
import jieba
#构造字典和weight矩阵
list_word=['UNK']
dict_word={}
tain_data_l_n=[]#左边LSTM的输入
tain_data_r_n=[]#右边LSTM的输入

for data in [tain_data_l,tain_data_r]:
    for line in data:
        words=list(jieba.cut(line))
        for i,word in enumerate(words):
            if word not in dict_word:
                dict_word[word]=len(dict_word)
print(dict_word)#字典构造完毕
id2w={dict_word[w]:w for w in dict_word}#word的索引
embedding_size=256
embedding_arry=np.random.randn(len(dict_word)+1,embedding_size)#句子embedding矩阵
embedding_arry[0]=0
word2vector=gensim.models.Word2Vec.load(path_word2vec)
for index,word in enumerate(dict_word):
    if word in word2vector.wv.vocab:
        embedding_arry[index]=word2vector.wv.word_vec(word)
print('embedding_arry shape:',embedding_arry.shape)
del word2vector
#将词组替换为索引
for line in tain_data_l:
    words = list(jieba.cut(line))
    for i,word in enumerate(words):
        words[i]=dict_word[word]
    tain_data_l_n.append(words)
print('tain_data_l_n length:',len(tain_data_l_n))
y_train=np.ones((len(tain_data_l_n),))
for line in tain_data_r:
    words = list(jieba.cut(line))
    for i,word in enumerate(words):
        words[i]=dict_word[word]
    tain_data_r_n.append(words)
print('tain_data_r_n length:',len(tain_data_r_n))
#得到语料中句子的最大长度
max_length=0
for line in tain_data_r_n:
    if max_length<len(line):
        max_length=len(line)
print('max length:',max_length)

# 对齐语料中句子的长度
tain_data_l_n = pad_sequences(tain_data_l_n, maxlen=max_length)
tain_data_r_n = pad_sequences(tain_data_r_n, maxlen=max_length)

print(tain_data_l_n)
print(tain_data_r_n)

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 5
n_epoch = 15

#相似度计算
def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


#输入层
left_input = Input(shape=(max_length,), dtype='int32')
right_input = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(len(embedding_arry), embedding_size, weights=[embedding_arry], input_length=max_length,
                            trainable=False)

#对句子embedding
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

#两个LSTM共享参数
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# model
malstm = Model([left_input, right_input], [malstm_distance])

optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
#train
malstm.fit(x=[np.asarray(tain_data_l_n), np.asarray(tain_data_r_n)], y=y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([np.asarray(tain_data_l_n), np.asarray(tain_data_r_n)], y_train) )

print(malstm.predict([np.array([1,2,3,4,5,6,7,7,0,0,0,0]).reshape(-1,12),np.array([1,2,23,4,12,6,7,22,0,0,0,0]).reshape(-1,12)]))




