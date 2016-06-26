# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import theano
import numpy as np
import theano.tensor as T

__author__ = 'fuhuamosi'


class HiddenLayer:
    def __init__(self, rng, x, n_in, n_out,
                 w=None, b=None, activation=T.tanh):
        self.x = x
        if w is None:
            w_values = np.asarray(rng.uniform(
                low=-np.sqrt(6 / (n_in + n_out)),
                high=np.sqrt(6 / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == T.nnet.sigmoid:
                w_values *= 4
            w = theano.shared(w_values, name='w', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(b_values, name='b', borrow=True)
        self.w = w
        self.b = b
        lin_output = T.dot(x, self.w) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.w, self.b]
