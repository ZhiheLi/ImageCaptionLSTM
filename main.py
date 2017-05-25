#coding: utf-8

from Preprocess import Preprocess
from build_model import BuildModel
import pickle as pkl


pp = Preprocess()
cnnDim, chrNum, cap_max_len, train_sample, valid_sample = pp.preprocess()
ic = BuildModel(cnnDim, chrNum, cap_max_len, train_sample, valid_sample)
pkl.dump([cnnDim, chrNum, cap_max_len, train_sample, valid_sample], open('preprocess.pkl', 'wb'))

ic.model_gen(lr = 0.1, dropout=0.25, embeddingDim=512, regularCoeff=0.001)
ic.model_fit(batch_size=512, epochs=1000)

ic.cap_gen(beam_size=20, index=9000)
