#coding: utf-8

from Preprocess import Preprocess
from build_model import BuildModel


pp = Preprocess()
cnnDim, chrNum, cap_max_len, train_sample, valid_sample = pp.preprocess()
t = BuildModel(cnnDim, chrNum, cap_max_len, train_sample, valid_sample)

t.model_gen(lr = 0.001, embeddingDim=1024, denseNode= 512, lstmNode=1000)
t.model_fit(batch_size=256, epochs=20, earlystop=10)

#t.model.load_weights('model_weights.h5')
t.cap_gen(beam_size=20, index=9000)
