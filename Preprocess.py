# coding: utf-8

"""Image caption data preprocessing."""

import pickle as pkl
import h5py
import numpy as np


class Preprocess():

    def __init__(self):
        self.index = 0
        self.cap_max_len = 0
        self.train_sample = 0
        self.valid_sample = 0


    def preprocess(self):
        """Preprocess captions, and build a word encoding dictionary."""
        index = 0
        dct = {'#': 0, '$': 1}

        train = []
        f = open('train.txt', encoding='utf-8')
        self.extract(train, f, dct, iftrain=True)
        f.close()

        pkl.dump(dct, open('dictionary.pkl', 'wb'))

        valid = []
        f = open('valid.txt', encoding='utf-8')
        self.extract(valid, f, {}, iftrain=False)
        f.close()

        pkl.dump(train, open('train.pkl', 'wb'))
        pkl.dump(valid, open('validation.pkl', 'wb'))
        
        f = h5py.File('image_vgg19_fc2_feature.h5', 'r')
        key = [key for key in f.keys()]
        cnn = f[key[1]].value
        return cnn[0].size, len(dct), self.cap_max_len+2, self.train_sample, self.valid_sample


    def extract(self, lst, f, dct, iftrain):
        """Helper to extract and preprocess captions from original caption data files."""
        sublst = []
        line = f.readline()
        for line in f:
            if str(self.index + 2) == line[:-1]:
                lst.append(sublst)
                self.index += 1
                sublst = []
            else:
                tmp = line[:-1]
                for c in tmp:
                    if c not in dct:
                        dct[c] = len(dct)
                self.cap_max_len = len(tmp) if len(tmp)>self.cap_max_len else self.cap_max_len
                if tmp:
                    sublst.append('#'+tmp+'$')    # starting and ending signs

                if iftrain:
                    self.train_sample += 1
                else:
                    self.valid_sample += 1

        lst.append(sublst)
        self.index += 1
        return
