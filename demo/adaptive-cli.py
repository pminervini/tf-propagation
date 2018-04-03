#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

from propagation.models import GaussianFields
from propagation.solvers import ExactSolver, JacobiSolver

from propagation.visualization import HintonDiagram

import logging


def main(argv):
    tf.set_random_seed(0)

    nb_rows, nb_cols = 6, 6
    nb_nodes = nb_rows * nb_cols

    edges_horizontal = [[(i, j), (i, j + 1)]
                      for i in range(nb_rows) for j in range(nb_cols)
                      if i < nb_rows and j < nb_cols - 1]

    edges_vertical = [[(i, j), (i + 1, j)]
                        for i in range(nb_rows) for j in range(nb_cols)
                        if i < nb_rows - 1 and j < nb_cols]

    edges = [(e, 0) for e in edges_horizontal] + [(e, 1) for e in edges_vertical]

    # We do not have an adjacency matrix, but rather a multi-relational adjacency tensor
    T = np.zeros(shape=[nb_nodes, nb_nodes, 2], dtype='float32')

    for ([(i, j), (k, l)], g) in edges:
        row, col = i * nb_rows + j, k * nb_cols + l
        T[row, col, g] = T[col, row, g] = 1.0

    l = np.zeros(shape=[nb_rows, nb_cols], dtype='int8')
    y = np.zeros(shape=[nb_rows, nb_cols], dtype='float32')

    l[0, 0] = 1
    y[0, 0] = 1.0

    l[0, nb_cols - 1] = 1
    y[0, nb_cols - 1] = 1.0

    l[nb_rows - 1, 0] = 1
    y[nb_rows - 1, 0] = - 1.0

    l[nb_rows - 1, nb_cols - 1] = 1
    y[nb_rows - 1, nb_cols - 1] = - 1.0

    mu, eps = 1.0, 1e-2

    batch_l = np.zeros(shape=[1, nb_nodes], dtype='float32')
    batch_y = np.zeros(shape=[1, nb_nodes], dtype='float32')
    batch_T = np.zeros(shape=[1, nb_nodes, nb_nodes, T.shape[2]], dtype='float32')

    batch_l[0, :] = l.reshape(nb_nodes)
    batch_y[0, :] = y.reshape(nb_nodes)
    batch_T[0, :, :, :] = T

    l_ph = tf.placeholder('float32', shape=[None, None], name='l')
    y_ph = tf.placeholder('float32', shape=[None, None], name='y')

    mu_ph = tf.placeholder('float32', [None], name='mu')
    eps_ph = tf.placeholder('float32', [None], name='eps')

    alpha = tf.get_variable('alpha', shape=[T.shape[2]],
                            initializer=tf.contrib.layers.xavier_initializer())

    T_ph = tf.placeholder('float32', shape=[None, None, None, None], name='T')

    W = tf.einsum('a,bmna->bmn', alpha, T_ph)

    solver = ExactSolver()
    # solver = JacobiSolver()

    l_idxs = tf.where(l_ph > 0)

    def leave_one_out_loss(a, x):
        idx_int = tf.cast(x, tf.int32)
        row_idx = tf.cast(l_idxs[idx_int, 0], tf.int32)
        col_idx = tf.cast(l_idxs[idx_int, 1], tf.int32)

        mask = tf.sparse_to_dense(
            sparse_indices=[[row_idx, col_idx]],
            output_shape=tf.shape(l_ph),
            sparse_values=0,
            default_value=1)
        mask = tf.cast(mask, tf.float32)

        mask = tf.nn.dropout(mask, keep_prob=0.5)

        model = GaussianFields(l=l_ph * mask, y=y_ph, mu=mu_ph, W=W, eps=eps_ph,
                               solver=solver)
        f_star = model.minimize()

        f_value = tf.cast(f_star[row_idx, col_idx], tf.float32)
        y_value = tf.cast(y_ph[row_idx, col_idx], tf.float32)

        res = a + (f_value - y_value) ** 2.0
        return res

    elems = tf.range(tf.shape(l_idxs)[0])
    elems = tf.identity(elems)

    initializer = tf.constant(0.0, dtype=tf.float32)

    loo_losses = tf.scan(lambda a, x: leave_one_out_loss(a, x),
                         elems=elems, initializer=initializer)
    loo_loss = loo_losses[-1]

    regularized_loo_loss = loo_loss + 0.1 * tf.nn.l2_loss(W)

    inf_model = GaussianFields(l=l_ph, y=y_ph, mu=mu_ph, W=W, eps=eps_ph, solver=solver)
    inf_f_star = inf_model.minimize()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(regularized_loo_loss, var_list=[alpha])

    init_op = tf.global_variables_initializer()

    feed_dict = {
        l_ph: batch_l,
        y_ph: batch_y,
        T_ph: batch_T,
        mu_ph: np.array([mu]),
        eps_ph: np.array([eps]),
    }

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        session.run(init_op)

        for i in range(1024):
            session.run(train_op, feed_dict=feed_dict)

            loo_loss_value = session.run(loo_loss, feed_dict=feed_dict)
            alpha_value = session.run(alpha, feed_dict=feed_dict)

            # print(loo_loss_value, alpha_value)

            hd = HintonDiagram()
            inf_f_value = session.run(inf_f_star, feed_dict=feed_dict)
            print(hd(inf_f_value[0, :].reshape((nb_rows, nb_cols))))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
