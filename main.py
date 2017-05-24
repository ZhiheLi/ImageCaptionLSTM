#coding: utf-8

from Preprocess import Preprocess
from build_model import BuildModel
import pickle as pkl


pp = Preprocess()
cnnDim, chrNum, cap_max_len, train_sample, valid_sample = pp.preprocess()
ic = BuildModel(cnnDim, chrNum, cap_max_len, train_sample, valid_sample)
pkl.dump([cnnDim, chrNum, cap_max_len, train_sample, valid_sample], open('preprocess.pkl', 'wb'))

ic.model_gen(lr = 0.01, dropout=0.25, embeddingDim=512, regularCoeff=0.001)
ic.model_fit(batch_size=256, epochs=50)

ic.cap_gen(beam_size=5, index=9000)
