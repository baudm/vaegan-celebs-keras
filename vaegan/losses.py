#!/usr/bin/env python3

import numpy as np

from keras import backend as K


def mean_gaussian_negative_log_likelihood(y_true, y_pred):
    c = 0.5 * np.log(2 * np.pi)
    nll = c + 0.5 * K.square(y_pred - y_true)
    return K.mean(nll)
