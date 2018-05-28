#!/usr/bin/env python3

import numpy as np

from keras import backend as K


def mean_gaussian_negative_log_likelihood(y_true, y_pred):
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
    axis = tuple(range(1, len(K.int_shape(y_true))))
    return K.mean(K.sum(nll, axis=axis), axis=-1)
