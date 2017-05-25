#coding: utf-8

from build_model import BuildModel
import pickle as pkl

[cnnDim, chrNum, cap_max_len, train_sample, valid_sample] = pkl.load(open('preprocess.pkl', 'rb'))
ic = BuildModel(cnnDim, chrNum, cap_max_len, train_sample, valid_sample)
ic.model_gen(lr = 0.01, dropout=0.25, embeddingDim=512, regularCoeff=0.001)

ic.cap_gen(beam_size=20, index=9000)
