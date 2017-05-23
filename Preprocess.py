# coding: utf-8

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
        index = 0
        table = {'0': '零', '1': '一', '2': '两', '3': '三', '4': '四',
                 '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
        dct = {'0': 0, '1': 1}

        train = []
        f = open('train.txt', encoding='utf-8')
        self.extract(train, f, table, dct, iftrain=True)
        f.close()

        valid = []
        f = open('valid.txt', encoding='utf-8')
        self.extract(valid, f, table, dct, iftrain=False)
        f.close()

        pkl.dump(train, open('train.pkl', 'wb'))
        pkl.dump(valid, open('validation.pkl', 'wb'))
        pkl.dump(dct, open('dictionary.pkl', 'wb'))

        f = h5py.File('image_vgg19_fc2_feature.h5', 'r')
        key = [key for key in f.keys()]
        cnn = f[key[1]].value
        return cnn[0].size, len(dct), self.cap_max_len+2, self.train_sample, self.valid_sample

    def extract(self, lst, f, table, dct, iftrain):
        sublst = []
        line = f.readline()
        for line in f:
            if str(self.index + 2) == line[:-1]:
                lst.append(sublst)
                self.index += 1
                sublst = []
            else:
                tmp = self.num2chn(line[:-1], table, dct)
                self.cap_max_len = len(tmp) if len(tmp)>self.cap_max_len else self.cap_max_len
                if tmp:
                    sublst.append('0'+tmp+'1')    #起始和终止字的选择

                if iftrain:
                    self.train_sample += 1
                else:
                    self.valid_sample += 1

        lst.append(sublst)
        self. index += 1
        return

    def num2chn(self, lst, table, dct):
        for i in range(0, len(lst)):
            if lst[i].isdigit():
                lst = lst.replace(lst[i], table[lst[i]])
            if ord(lst[i]) < 256:   # 是否需要删除英文字母和标点？（其实这么处理后还是有汉字标点）
                return ''
            elif lst[i] not in dct:
                dct[lst[i]] = len(dct)
        return lst


