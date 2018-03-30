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

    l_ph = tf.placeholder('float32', shape=[None, None], name='l')
    y_ph = tf.placeholder('float32', shape=[None, None], name='y')

    mu_ph = tf.placeholder('float32', [None], name='mu')
    eps_ph = tf.placeholder('float32', [None], name='eps')

    W_ph = tf.placeholder('float32', shape=[None, None, None], name='W')

    with tf.Session() as session:
        solver = ExactSolver()

        feed_dict = {
            l_ph: l.reshape(1, N),
            y_ph: y.reshape(1, N),
            W_ph: W.reshape(1, N, N),
            mu_ph: np.array([mu] * 1),
            eps_ph: np.array([eps] * 1),
        }

        model = GaussianFields(l_ph, y_ph, mu_ph, W_ph, eps_ph, solver=solver)

        f = model.minimize()

        f_value = session.run(f, feed_dict=feed_dict)

        assert f_value[0, 1] > 0
        assert f_value[0, N - 2] < 0


def test_minimize():
    nb_rows, nb_cols = 40, 40
    nb_nodes = nb_rows * nb_cols

    edges = []

    edges += [[(i, j), (i, j + 1)]
              for i in range(nb_rows) for j in range(nb_cols)
              if i < nb_rows and j < nb_cols - 1]

    edges += [[(i, j), (i + 1, j)]
              for i in range(nb_rows) for j in range(nb_cols)
              if i < nb_rows - 1 and j < nb_cols]

    # edges += [[(i, j), (i + 1, j + 1)]
    #           for i in range(nb_rows) for j in range(nb_cols)
    #           if i < nb_rows - 1 and j < nb_cols - 1]

    W = np.zeros(shape=[nb_nodes, nb_nodes], dtype='float32')

    for [(i, j), (k, l)] in edges:
        row, col = i * nb_rows + j, k * nb_cols + l
        W[row, col] = W[col, row] = 1

    mu, eps = 1.0, 1e-8

    l_ph = tf.placeholder('float32', shape=[None, None], name='l')
    y_ph = tf.placeholder('float32', shape=[None, None], name='y')

    mu_ph = tf.placeholder('float32', [None], name='mu')
    eps_ph = tf.placeholder('float32', [None], name='eps')

    W_ph = tf.placeholder('float32', shape=[None, None, None], name='W')

    f_ph = tf.placeholder('float32', shape=[None, None], name='f')

    solver = ExactSolver()
    model = GaussianFields(l_ph, y_ph, mu_ph, W_ph, eps_ph, solver=solver)
    e = model(f_ph)

    f_star = model.minimize()

    rs = np.random.RandomState(0)

    with tf.Session() as session:

        for _ in range(8):
            l = np.zeros(shape=[nb_rows, nb_cols], dtype='int8')
            y = np.zeros(shape=[nb_rows, nb_cols], dtype='float32')

            for i in range(l.shape[0]):
                for j in range(l.shape[1]):
                    l[i, j] = rs.randint(2)
                    y[i, j] = rs.rand()

            batch_l = np.zeros(shape=[2, nb_nodes])
            batch_y = np.zeros(shape=[2, nb_nodes])
            batch_W = np.zeros(shape=[2, nb_nodes, nb_nodes])

            batch_l[0, :] = l.reshape(nb_nodes)
            batch_y[0, :] = y.reshape(nb_nodes)
            batch_W[0, :, :] = W

            batch_l[1, :] = l.reshape(nb_nodes)
            batch_y[1, :] = y.reshape(nb_nodes)
            batch_W[1, :, :] = - W

            feed_dict = {
                l_ph: batch_l,
                y_ph: batch_y,
                W_ph: batch_W,
                mu_ph: np.array([mu] * 2),
                eps_ph: np.array([eps] * 2)
            }

            f_value = session.run(f_star, feed_dict=feed_dict)

            feed_dict = {
                l_ph: batch_l,
                y_ph: batch_y,
                W_ph: batch_W,
                mu_ph: np.array([mu] * 2),
                eps_ph: np.array([eps] * 2),
                f_ph: f_value
            }

            minimum_e_value = session.run(e, feed_dict=feed_dict)

            for _ in range(8):
                new_f_value = np.copy(f_value)
                for i in range(f_value.shape[0]):
                    for j in range(f_value.shape[1]):
                        new_f_value[i, j] += rs.normal(0.0, 0.1)

                feed_dict = {
                    l_ph: batch_l,
                    y_ph: batch_y,
                    W_ph: batch_W,
                    mu_ph: np.array([mu] * 2),
                    eps_ph: np.array([eps] * 2),
                    f_ph: new_f_value
                }

                new_e_value = session.run(e, feed_dict=feed_dict)

                for i in range(minimum_e_value.shape[0]):
                    assert minimum_e_value[i] <= new_e_value[i]


if __name__ == '__main__':
    pytest.main([__file__])
