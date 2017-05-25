# coding: utf-8

from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, Merge, Layer, Reshape, Lambda, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras import metrics, optimizers
from keras.callbacks import EarlyStopping
from keras import backend
from keras.utils import plot_model
from keras import regularizers

import tensorflow as tf
import numpy as np
import pickle as pkl
import h5py
import copy
import random
import os
import time

def slice(x):
        return tf.slice(x, [0,1,0],[-1,-1,-1])

class BuildModel():

    def __init__(self, cnnDim, chrNum, cap_max_len, train_sample, valid_sample):
        self.cnnDim = cnnDim                #CNN特征维度
        self.chrNum = chrNum                #汉字总数量
        self.cap_max_len = cap_max_len      #最大句长
        self.train_sample = train_sample    #训练集总句子数
        self.valid_sample = valid_sample    #验证集总句子数
        self.model = None

        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        backend.set_session(sess)

    def img_extract(self, key):
        f = h5py.File('image_vgg19_fc2_feature.h5', 'r')
        return f[key].value

    def cap_extract(self, type):
        txt = pkl.load(open('%s.pkl' % type, 'rb'))
        dct = pkl.load(open('dictionary.pkl', 'rb'))

        cap = []
        for lst in txt:
            tmp = []
            for s in lst:
                tmp.append([dct[k] for k in s if k in dct])
            cap.append(tmp)
        return cap

    def data_gen(self, key):
        img_all_ori = self.img_extract(key+'_set')
        cap_all_ori = self.cap_extract(key)

        img = []
        next_word = []
        word_seq = []
        rand_index = random.sample(range(len(cap_all_ori)), len(cap_all_ori))
        img_all = img_all_ori[rand_index]
        cap_all = np.array(cap_all_ori)[rand_index]

        for i, lst in enumerate(cap_all):
            for s in lst:
                next_word.append(s[1:])
                word_seq.append(s[:-1])
                img.append(img_all[i])
            

        img = np.array(img)
        next_word = np.array(next_word)
        next_word = pad_sequences(next_word, maxlen=self.cap_max_len, padding='post', value=1)
        next_word = np.expand_dims(next_word, -1)
        word_seq = pad_sequences(word_seq, maxlen=self.cap_max_len, padding='post', value=1)
        print(img.shape)
        print(next_word.shape)
        print(word_seq.shape)
        return ([img, word_seq], next_word)
                            
    def model_gen(self, lr, dropout, embeddingDim, regularCoeff):
        img_mdl = Sequential()
        img_mdl.add(Dense(embeddingDim, activation='relu', input_dim=self.cnnDim, kernel_regularizer=regularizers.l2(2*regularCoeff), bias_regularizer=regularizers.l2(regularCoeff)))
        img_mdl.add(Reshape((1,embeddingDim)))
        
        cap_mdl = Sequential()
        cap_mdl.add(Embedding(input_dim=self.chrNum, output_dim=embeddingDim, input_length=self.cap_max_len))

        self.model = Sequential()
        self.model.add(Merge([img_mdl, cap_mdl], mode='concat', concat_axis=1))
        self.model.add(LSTM(embeddingDim, return_sequences=True, dropout=dropout, kernel_regularizer=regularizers.l2(2*regularCoeff), bias_regularizer=regularizers.l2(regularCoeff), recurrent_initializer='zeros'))
        self.model.add(Lambda(slice))
        self.model.add(TimeDistributed(Dense(self.chrNum, activation='softmax')))

        sgd = optimizers.SGD(lr = lr)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,
                           metrics=[metrics.sparse_categorical_accuracy])

        plot_model(self.model, to_file='model.png', show_shapes=True)
        return

    def model_fit(self, batch_size, epochs):
                
        (trainX, trainY) = self.data_gen('train') 
        (valX, valY) = self.data_gen('validation')
        self.model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(valX, valY))

        self.model.save('model_weights.h5')
        return

    def cap_gen(self, beam_size, index):
        self.model = load_model('my_model.h5')
        img_all = self.img_extract('test_set')
        cap_pred = []
        dct = pkl.load(open('dictionary.pkl', 'rb'))
        map = self.map2word(dct)
        f = open('test.txt', 'w')

        for img in img_all:
            cap_beam = [[[dct['#']], 0.0]] * beam_size
            end = False
            count = 0
            while not end and count < self.cap_max_len:
                cap_cand = []
                itr = 1 if count == 0 else beam_size
                for i in range(itr):
                    if cap_beam[i][0][-1] != dct['$']:
                        cap_pad = np.ones((self.cap_max_len,))
                        cap_pad[:len(cap_beam[i][0])] = cap_beam[i][0]
                        prob = np.log(self.model.predict([np.array([img]), np.array([cap_pad])])[0][count][:])
                        next_word_ind = np.argsort(prob)[-beam_size:]
                        cap_tmp = self.nextWordAppend(cap_beam[i], next_word_ind, prob[next_word_ind], beam_size)
                    else:
                        cap_tmp = [cap_beam[i]]
                    cap_cand.extend(cap_tmp)
                cap_beam = sorted(cap_cand, key=lambda x: x[1])[-beam_size:]
                end = self.ifend(cap_beam, beam_size, dct)
                count += 1
            cap = max(cap_beam, key=lambda x: x[1])[0]
            cap, index = self.ind2word(cap, map, index)
            cap_pred.append(cap)
            f.write(cap)
            print('Image %d Captioned' % index)
            print(cap)
        f.close()
        return

    def nextWordAppend(self, cap, next_word, prob, beam_size):
        out = []
        for i in range(beam_size):
            tmp = copy.deepcopy(cap)
            tmp[0].append(next_word[i])
            tmp[1] += prob[i]
            out.append(tmp)
        return out

    def ifend(self, cap_beam, beam_size, dct):
        flag = True
        for i in range(beam_size):
            flag &= (cap_beam[0][-1] == dct['$'])
        return flag

    def ind2word(self, cap, map, index):
        s = []
        for n in cap[1:-1]:
            s.append(map[n])
        out = str(index) + '\n' + ''.join(s) + '\n'
        index += 1
        return out, index

    def map2word(self, dct):
        out = [''] * len(dct)
        for k in dct:
            out[dct[k]] = k
        return out

