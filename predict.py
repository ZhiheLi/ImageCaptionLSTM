#coding: utf-8

from build_model import BuildModel
import pickle as pkl

[cnnDim, chrNum, cap_max_len, train_sample, valid_sample] = pkl.load(open('preprocess.pkl', 'rb'))
ic = BuildModel(cnnDim, chrNum, cap_max_len, train_sample, valid_sample)

ic.cap_gen(beam_size=1, index=9000)
