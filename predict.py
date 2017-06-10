#!/usr/bin/env python3
#coding: utf-8

"""Generate predicted captions for test set / validation set."""

from build_model import BuildModel
from sys import argv
import pickle as pkl

def main(dataset, prev_index):
    assert(dataset in ('test', 'validation'))
    [cnnDim, chrNum, cap_max_len, train_sample, valid_sample] = pkl.load(open('preprocess.pkl', 'rb'))
    ic = BuildModel(cnnDim, chrNum, cap_max_len, train_sample, valid_sample)
    ic.model_gen(lr=0.01, dropout=0.25, embeddingDim=512, regularCoeff=0.001)

    ic.cap_gen(beam_size=20, index=prev_index, dataset=dataset)


if __name__ == '__main__':
    # Usage: argv[0] [test/validation]
    dataset = 'test'
    prev_index = 9000
    if len(argv) >= 2:
        dataset = argv[1]
        prev_index = 8001

    main(dataset, prev_index)
