# -*- coding: utf-8 -*-

import pytest

import numpy as np
import tensorflow as tf

from propagation.solvers import ExactSolver
from propagation.models import GaussianFields


def test_model():
    rows, cols = 4, 4
    N = rows * cols

    edges = [[(i, j), (i, j + 1)] for i in range(rows) for j in range(cols) if i < rows and j < cols - 1]
    edges += [[(i, j), (i + 1, j)] for i in range(rows) for j in range(cols) if i < rows - 1 and j < cols]

    W = np.zeros((N, N))

    for [(i, j), (k, l)] in edges:
        row, col = i * rows + j, k * cols + l
        W[row, col] = W[col, row] = 1

    l, y = np.zeros(shape=[N,], dtype='int8'), np.zeros(shape=[N,])
    l[0], y[0] = 1, 1
    l[rows - 1], y[rows - 1] = 1, 1

    l[N - 1], y[N - 1] = 1, -1
    l[N - rows], y[N - rows] = 1, -1

    mu, eps = 1.0, 1e-8

    l_ph = tf.placeholder('float32', shape=[1, None], name='l')
    y_ph = tf.placeholder('float32', shape=[1, None], name='y')

    mu_ph = tf.placeholder('float32', None, name='mu')
    eps_ph = tf.placeholder('float32', None, name='eps')

    W_ph = tf.placeholder('float32', shape=[1, None, None], name='W')

    with tf.Session() as session:
        solver = ExactSolver()

        feed_dict = {
            l_ph: l.reshape(1, N),
            y_ph: y.reshape(1, N),
            W_ph: W.reshape(1, N, N),
            mu_ph: mu, eps_ph: eps
        }

        model = GaussianFields(l_ph, y_ph, mu_ph, W_ph, eps_ph, solver=solver)

        f = model.minimize()

        f_value = session.run(f, feed_dict=feed_dict)

        assert f_value[0, 1] > 0
        assert f_value[0, N - 2] < 0

if __name__ == '__main__':
    pytest.main([__file__])
