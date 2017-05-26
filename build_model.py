# coding: utf-8

from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, Merge, Layer, Reshape, Lambda, TimeDistributed, Activation
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

def _loss(y_true, y_pred):
    epsilon = tf.convert_to_tensor(10e-8)
    if epsilon.dtype != y_pred.dtype.base_dtype:
        epsilon = tf.cast(epsilon, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.log(y_pred)

    output_shape = y_pred.get_shape()
    label = tf.slice(y_true, [0,0,0], [-1,-1,1])
    mask = tf.slice(y_true, [0,0,1], [-1,-1,1])
    label = backend.cast(backend.flatten(label), 'int64')
    mask = backend.flatten(mask)
    logits = tf.reshape(y_pred, [-1, int(output_shape[-1])])
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=logits)
    res = tf.reshape(res * mask, tf.shape(y_pred)[:-1])
    return backend.mean(res, axis=-1)


class BuildModel():

    def __init__(self, cnnDim, chrNum, cap_max_len, train_sample, valid_sample):
        self.cnnDim = cnnDim                #CNN特征维度
        self.chrNum = chrNum                #汉字总数量
        self.cap_max_len = cap_max_len      #最大句长
        self.train_sample = train_sample    #训练集总句子数
        self.valid_sample = valid_sample    #验证集总句子数
        self.model = None

        os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
        mask = []
        word_seq = []
        rand_index = random.sample(range(len(cap_all_ori)), len(cap_all_ori))
        img_all = img_all_ori[rand_index]
        cap_all = np.array(cap_all_ori)[rand_index]

        for i, lst in enumerate(cap_all):
            for s in lst:
                next_word.append(s[1:])
                mask.append([1] * (len(s)-1))
                word_seq.append(s[:-1])
                img.append(img_all[i])
            

        img = np.array(img)
        next_word = np.array(next_word)
        next_word = pad_sequences(next_word, maxlen=self.cap_max_len, padding='post', value=1)
        next_word = np.expand_dims(next_word, -1)
        mask = np.array(mask)
        mask = pad_sequences(mask, maxlen=self.cap_max_len, padding='post', value=0)
        mask = np.expand_dims(mask, -1)
        next_word = np.concatenate((next_word, mask), axis=-1)
        word_seq = pad_sequences(word_seq, maxlen=self.cap_max_len, padding='post', value=1)
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
        self.model.compile(loss=_loss, optimizer=sgd,
                           metrics=[metrics.sparse_categorical_accuracy])

        plot_model(self.model, to_file='model.png', show_shapes=True)
        return

    def model_fit(self, batch_size, epochs):
                
        (trainX, trainY) = self.data_gen('train') 
        (valX, valY) = self.data_gen('validation')
        self.model.load_weights('my_model.h5')
        self.model.fit(trainX, trainY, batch_size=batch_size, shuffle=True, epochs=epochs, validation_data=(valX, valY))

        self.model.save_weights('my_model.h5')
        return

    def cap_gen(self, beam_size, index):
        self.model.load_weights('my_model.h5')
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
            flag &= (cap_beam[i][0][-1] == dct['$'])
        return flag

    def ind2word(self, cap, map, index):
        s = []
        for n in cap[1:-1]:
            s.append(map[n])
        out = str(index) + ' ' + ' '.join(s) + '\n'
        index += 1
        return out, index

    def map2word(self, dct):
        out = [''] * len(dct)
        for k in dct:
            out[dct[k]] = k
        return out

