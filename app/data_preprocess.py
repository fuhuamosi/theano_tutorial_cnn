# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip, pickle
import theano
import numpy as np
import theano.tensor as T

__author__ = 'fuhuamosi'


def load_data():
    print('加载数据开始...')
    with gzip.open('../dataset/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, dtype='int32')
    train_x, train_y = shared_dataset(train_set)
    valid_x, valid_y = shared_dataset(valid_set)
    test_x, test_y = shared_dataset(test_set)
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('加载数据结束')
    return rval

