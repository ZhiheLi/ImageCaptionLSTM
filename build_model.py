# coding: utf-8

from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, Merge, TimeDistributed, RepeatVector
from keras.preprocessing.sequence import pad_sequences
from keras import metrics, optimizers
from keras.callbacks import EarlyStopping
from keras import backend
from keras.utils import plot_model

import tensorflow as tf
import numpy as np
import pickle as pkl
import h5py
import copy
import random
import os


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

    def data_gen(self, batch_size, key):
        img_all_ori = self.img_extract(key+'_set')
        cap_all_ori = self.cap_extract(key)

        count = 0
        img = []
        next_word = []
        word_seq = []
        while 1:
            rand_index = random.sample(range(len(cap_all_ori)), len(cap_all_ori))
            img_all = img_all_ori[rand_index]
            cap_all = np.array(cap_all_ori)[rand_index]

            for i, lst in enumerate(cap_all):
                for s in lst:
                    for j, c in enumerate(s[:-1]):
                        count += 1
                        next_word.append(np.zeros((self.chrNum,)))
                        next_word[-1][s[j+1]] = 1
                        word_seq.append(s[:j+1])
                        img.append(img_all[i])

                        if count == batch_size:
                            img = np.array(img)
                            next_word = np.array(next_word)
                            word_seq = pad_sequences(word_seq, maxlen=self.cap_max_len, padding='post')
                            yield [[img, word_seq], next_word]
                            count = 0
                            img = []
                            next_word = []
                            word_seq = []
                            
    def model_gen(self, lr, dropout, embeddingDim, denseNode, lstmNode):
        img_mdl = Sequential()
        img_mdl.add(Dense(denseNode, activation='relu', input_dim=self.cnnDim))
        img_mdl.add(RepeatVector(self.cap_max_len))

        cap_mdl = Sequential()
        cap_mdl.add(Embedding(input_dim=self.chrNum, output_dim=embeddingDim, input_length=self.cap_max_len))
        cap_mdl.add(LSTM(units=embeddingDim, return_sequences=True))       #这里为何要加LSTM
        cap_mdl.add(TimeDistributed(Dense(denseNode)))

        self.model = Sequential()
        self.model.add(Merge([img_mdl, cap_mdl], mode='concat'))
        self.model.add(LSTM(lstmNode, return_sequences=False, dropout=dropout))
        self.model.add(Dense(self.chrNum, activation='softmax'))

        rmsprop = optimizers.rmsprop(lr = lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop,
                           metrics=[metrics.categorical_accuracy])

        plot_model(self.model, to_file='model.png', show_shapes=True)
        return

    def model_fit(self, batch_size, epochs, earlystop):
        val = next(self.data_gen(batch_size=1000, key='validation'))
        es = EarlyStopping(monitor='val_loss', patience=earlystop)

        self.model.fit_generator(self.data_gen(batch_size=batch_size, key='train'),
                                 steps_per_epoch=int(self.train_sample/batch_size),
                                 epochs=epochs, verbose=1, callbacks=[es],
                                 validation_data=val)
        self.model.save_weights('model_weights.h5')
        return

    def cap_gen(self, beam_size, index):
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
                count += 1
                cap_cand = []
                itr = 1 if count == 1 else beam_size
                for i in range(itr):
                    if cap_beam[i][0][-1] != dct['$']:
                        cap_pad = np.zeros((self.cap_max_len,))
                        cap_pad[:len(cap_beam[i][0])] = cap_beam[i][0]
                        prob = np.log(self.model.predict([np.array([img]), np.array([cap_pad])])[0])
                        next_word_ind = np.argsort(prob)[-beam_size:]
                        cap_tmp = self.nextWordAppend(cap_beam[i], next_word_ind, prob, beam_size)
                    else:
                        cap_tmp = [cap_beam[i]]
                    cap_cand.extend(cap_tmp)
                cap_beam = sorted(cap_cand, key=lambda x: x[1])[-beam_size:]
                end = self.ifend(cap_beam, beam_size, dct)

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
            flag &= (cap_beam[0][-1] == dct['1'])
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

