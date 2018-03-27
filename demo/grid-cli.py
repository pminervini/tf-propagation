#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

from propagation.models import GaussianFields
from propagation.solvers import ExactSolver

from propagation.visualization import HintonDiagram

import logging


def main(argv):
    nb_rows, nb_cols = 40, 40
    nb_nodes = nb_rows * nb_cols

    edges = []

    edges += [[(i, j), (i, j + 1)]
              for i in range(nb_rows) for j in range(nb_cols)
              if i < nb_rows and j < nb_cols - 1]

    edges += [[(i, j), (i + 1, j)]
              for i in range(nb_rows) for j in range(nb_cols)
              if i < nb_rows - 1 and j < nb_cols]

    edges += [[(i, j), (i + 1, j + 1)]
              for i in range(nb_rows) for j in range(nb_cols)
              if i < nb_rows - 1 and j < nb_cols - 1]

    W = np.zeros((nb_nodes, nb_nodes))

    for [(i, j), (k, l)] in edges:
        row, col = i * nb_rows + j, k * nb_cols + l
        W[row, col] = W[col, row] = 1

    l, y = np.zeros(shape=[nb_nodes, ], dtype='int8'), np.zeros(shape=[nb_nodes, ], dtype='float32')
    l[0], y[0] = 1, 1.2
    # L[rows - 1], y[rows - 1] = 1, 1

    l[nb_nodes - 1], y[nb_nodes - 1] = 1, -1.0
    # L[N - rows], y[N - rows] = 1, -1

    for i in range(len(y)):
        if -1e-8 < y[i] < 1e-8:
            y[i] = 0

    mu, eps = 1.0, 1e-8

    l_ph = tf.placeholder('float32', shape=[None], name='l')
    y_ph = tf.placeholder('float32', shape=[None], name='y')

    mu_ph = tf.placeholder('float32', None, name='mu')
    eps_ph = tf.placeholder('float32', None, name='eps')

    W_ph = tf.placeholder('float32', shape=[None, None], name='W')

    with tf.Session() as session:
        solver = ExactSolver()

        feed_dict = {
            l_ph: l, y_ph: y,
            mu_ph: mu, eps_ph: eps,
            W_ph: W
        }

        model = GaussianFields(l_ph, y_ph, mu_ph, W_ph, eps_ph, solver=solver)

        f = model.minimize()

        hd = HintonDiagram()

        f_value = session.run(f, feed_dict=feed_dict)

        print(hd(f_value.reshape((nb_rows, nb_cols))))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
