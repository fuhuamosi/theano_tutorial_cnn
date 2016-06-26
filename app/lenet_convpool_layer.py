# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

__author__ = 'fuhuamosi'


class LeNetConvPoolLayer:
    def __init__(self, rng, input_data, filter_shape, input_shape,
                 pool_size=(2, 2)):
        """
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input_data: theano.tensor.dtensor4
            :param input_data: symbolic image tensor, of shape image_shape

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height, filter width)

            :type input_shape: tuple or list of length 4
            :param input_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type pool_size: tuple or list of length 2
            :param pool_size: the downsampling (pooling) factor (#rows, #cols)
            """
        assert filter_shape[1] == input_shape[1]
        self.input_data = input_data
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(input_shape[2:])) // (np.prod(pool_size))
        w_bound = np.sqrt(6 / (fan_in + fan_out))
        self.w = theano.shared(np.asarray(rng.uniform(low=-1.0 / w_bound,
                                                      high=1.0 / w_bound,
                                                      size=filter_shape),
                                          dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((filter_shape[0],),
                               dtype=theano.config.floatX), borrow=True)
        conv_out = T.nnet.conv2d(input=input_data, filters=self.w,
                                 input_shape=input_shape, filter_shape=filter_shape)
        pool_out = pool_2d(input=conv_out, ds=pool_size, ignore_border=True)
        self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.w, self.b]
