# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from app.data_preprocess import load_data
import theano.tensor as T
import numpy as np
from app.lenet_convpool_layer import LeNetConvPoolLayer
from app.hidden_layer import HiddenLayer
from app.log_regression import LogisticRegression
import theano
import timeit
import os
import sys

__author__ = 'fuhuamosi'


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, nkerns=None, batch_size=500):
    if nkerns is None:
        nkerns = [20, 50]
    dataset = load_data()
    train_x, train_y = dataset[0]
    valid_x, valid_y = dataset[1]
    test_x, test_y = dataset[2]
    n_train_batches = train_x.get_value().shape[0] // batch_size
    n_valid_batches = valid_x.get_value().shape[0] // batch_size
    n_test_batches = test_x.get_value().shape[0] // batch_size

    # 建立模型
    print('建立模型开始...')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(23455)
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(rng=rng, input_data=layer0_input,
                                filter_shape=(nkerns[0], 1, 5, 5),
                                input_shape=(batch_size, 1, 28, 28))
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(rng=rng, input_data=layer0.output,
                                filter_shape=(nkerns[1], nkerns[0], 5, 5),
                                input_shape=(batch_size, nkerns[0], 12, 12))
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(rng=rng, x=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)
    layer3 = LogisticRegression(x=layer2.output, n_in=500, n_out=10)
    cost = layer3.negative_log_likelihood(y)

    test_model = theano.function(inputs=[index],
                                 outputs=layer3.errors(y),
                                 givens={x: test_x[index * batch_size: (index + 1) * batch_size],
                                         y: test_y[index * batch_size: (index + 1) * batch_size],
                                         })
    valid_model = theano.function(inputs=[index],
                                  outputs=layer3.errors(y),
                                  givens={x: valid_x[index * batch_size: (index + 1) * batch_size],
                                          y: valid_y[index * batch_size: (index + 1) * batch_size],
                                          })
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, params)
    updates = [(param, param - learning_rate * grad)
               for param, grad in zip(params, grads)]
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={x: train_x[index * batch_size: (index + 1) * batch_size],
                                          y: train_y[index * batch_size: (index + 1) * batch_size],
                                          })
    print('建立模型结束')
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):

            iteration = (epoch - 1) * n_train_batches + minibatch_index

            if iteration % 100 == 0:
                print('training @ iter = ', iteration)
            cost_ij = train_model(minibatch_index)

            if (iteration + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [valid_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iteration * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iteration

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iteration:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    evaluate_lenet5()
